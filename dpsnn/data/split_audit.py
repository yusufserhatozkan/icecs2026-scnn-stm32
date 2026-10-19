"""Fail-closed provenance checks for corrected VoiceBank experiments.

Preparation audits all splits. Fitting can verify train/validation using the
frozen report without accessing any official-test manifest, cache or waveform.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COUNTS = {"train": 10802, "valid": 770, "test": 824}
VALID_SPEAKERS = {"p226", "p287"}
TEST_SPEAKERS = {"p232", "p257"}


class SplitAuditError(ValueError):
    """The supplied data do not match the frozen experiment provenance."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(path, repository_root=REPOSITORY_ROOT):
    root = Path(repository_root).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root):
        raise SplitAuditError(f"Path is outside the repository: {path}")
    return candidate


def relative_path(path, repository_root=REPOSITORY_ROOT):
    return repo_path(path, repository_root).relative_to(Path(repository_root).resolve()).as_posix()


def file_record(path, repository_root=REPOSITORY_ROOT):
    path = repo_path(path, repository_root)
    return {"path": relative_path(path, repository_root),
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def audit_manifest(csv_path, expected_count, expected_speakers=None,
                   repository_root=REPOSITORY_ROOT, sample_rate=16000, channels=1):
    """Check every stored waveform path and clean/noisy utterance identity."""
    path = repo_path(csv_path, repository_root)
    import soundfile as sf

    ids, frames = [], []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"ID", "duration", "noisy_wav", "clean_wav"}
        if not required.issubset(reader.fieldnames or []):
            raise SplitAuditError(f"Missing manifest columns: {path.name}")
        for row in reader:
            utterance_id = row["ID"]
            if not utterance_id or "_" not in utterance_id:
                raise SplitAuditError(f"Invalid utterance ID: {utterance_id!r}")
            try:
                duration = float(row["duration"])
            except (ValueError, TypeError) as exc:
                raise SplitAuditError(f"Invalid duration for {utterance_id}") from exc
            if not math.isfinite(duration) or duration <= 0:
                raise SplitAuditError(f"Invalid duration for {utterance_id}")
            paths = [repo_path(row[key], repository_root)
                     for key in ("noisy_wav", "clean_wav")]
            if paths[0] == paths[1]:
                raise SplitAuditError(f"Same clean/noisy source for {utterance_id}")
            for wav_path in paths:
                if not wav_path.is_file() or wav_path.stem != utterance_id:
                    raise SplitAuditError(f"Missing or mismatched waveform for {utterance_id}: {wav_path}")
            audio_info = [sf.info(str(wav_path)) for wav_path in paths]
            if any(info.samplerate != sample_rate or info.channels != channels for info in audio_info):
                raise SplitAuditError(f"Manifest waveform rate/channel mismatch: {utterance_id}")
            if audio_info[0].frames != audio_info[1].frames or abs(duration - audio_info[0].frames/sample_rate) > 1e-6:
                raise SplitAuditError(f"Manifest waveform length/duration mismatch: {utterance_id}")
            ids.append(utterance_id)
            frames.append(audio_info[0].frames)
    if len(ids) != len(set(ids)):
        raise SplitAuditError(f"Duplicate utterance IDs in {path.name}")
    if len(ids) != expected_count:
        raise SplitAuditError(f"Manifest count {len(ids)} != {expected_count}: {path.name}")
    speakers = sorted({item.split("_", 1)[0] for item in ids})
    if expected_speakers is not None and set(speakers) != set(expected_speakers):
        raise SplitAuditError(f"Manifest speaker mismatch: {path.name}")
    return {**file_record(path, repository_root), "count": len(ids),
            "ids": ids, "speakers": speakers, "frames": frames}


def audit_hdf5_cache(hdf5_path, manifest_audit, sample_rate=16000, channels=1,
                     repository_root=REPOSITORY_ROOT):
    """Validate cache metadata and ordered IDs; hashing covers waveform values."""
    import h5py
    import numpy as np

    path = repo_path(hdf5_path, repository_root)
    ids = []
    try:
        with h5py.File(path, "r") as cache:
            if cache.attrs.get("sr") != sample_rate or cache.attrs.get("channels") != channels:
                raise SplitAuditError(f"Cache rate/channel mismatch: {path.name}")
            if len(cache) != manifest_audit["count"]:
                raise SplitAuditError(f"Cache count mismatch: {path.name}")
            if set(cache.keys()) != {str(i) for i in range(len(cache))}:
                raise SplitAuditError(f"Invalid cache group keys: {path.name}")
            for index in range(len(cache)):
                group = cache[str(index)]
                utterance_id = group.attrs["ID"]
                if isinstance(utterance_id, bytes):
                    utterance_id = utterance_id.decode("utf-8")
                ids.append(str(utterance_id))
                length = int(group.attrs["length"])
                if (length <= 0 or int(group.attrs["clean_length"]) != length
                        or length != manifest_audit["frames"][index]):
                    raise SplitAuditError(f"Cache length mismatch: {utterance_id}")
                for name in ("noisy", "clean"):
                    if group[name].shape != (channels, length) or group[name].dtype.name != "float32":
                        raise SplitAuditError(f"Cache shape/dtype mismatch: {utterance_id}/{name}")
                    if not np.isfinite(group[name][()]).all():
                        raise SplitAuditError(f"Nonfinite cache waveform: {utterance_id}/{name}")
    except (OSError, KeyError, TypeError) as exc:
        raise SplitAuditError(f"Unreadable or partial cache: {path.name}") from exc
    if ids != manifest_audit["ids"] or len(ids) != len(set(ids)):
        raise SplitAuditError(f"Cache IDs differ from manifest: {path.name}")
    return {**file_record(path, repository_root), "count": len(ids), "ids": ids,
            "speakers": sorted({item.split("_", 1)[0] for item in ids}),
            "sample_rate": sample_rate, "channels": channels}


def ensure_hdf5_cache(csv_path, hdf5_path, expected_audit, create_fn=None,
                      sample_rate=16000, channels=1, repository_root=REPOSITORY_ROOT):
    """Verify an existing cache or publish exactly one missing cache exclusively."""
    path = repo_path(hdf5_path, repository_root)
    if path.exists():
        return audit_hdf5_cache(path, expected_audit, sample_rate, channels, repository_root)
    if create_fn is None:
        from .hdf5_prepare import create_hdf5
        create_fn = create_hdf5
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        create_fn(str(repo_path(csv_path, repository_root)), str(temporary_path),
                  sample_rate, channels=channels)
        audit = audit_hdf5_cache(temporary_path, expected_audit, sample_rate, channels, repository_root)
        # A hard link is atomic and fails if another process has created path.
        # Both files are in the same directory/filesystem. Never use replace().
        os.link(temporary_path, path)
        return {**audit, "path": relative_path(path, repository_root)}
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_split_audits(splits, expected_counts=EXPECTED_COUNTS):
    """Validate all-split metadata without touching the referenced data paths."""
    if set(splits) != {"train", "valid", "test"}:
        raise SplitAuditError("Exactly train, valid and test split records are required")
    paths = []
    for name, split in splits.items():
        manifest, cache = split["manifest"], split["cache"]
        ids = manifest["ids"]
        speakers = {item.split("_", 1)[0] for item in ids}
        if len(ids) != len(set(ids)) or len(ids) != expected_counts[name]:
            raise SplitAuditError(f"Invalid frozen IDs/count: {name}")
        for record in (manifest, cache):
            if record["count"] != len(ids) or record["ids"] != ids or set(record["speakers"]) != speakers:
                raise SplitAuditError(f"Inconsistent frozen metadata: {name}")
            # Lexical check only: do not stat/resolve the official test path.
            record_path = Path(record["path"])
            if record_path.is_absolute() or ".." in record_path.parts:
                raise SplitAuditError(f"Frozen paths must be relative to repository: {name}")
            paths.append(os.path.normcase(os.path.normpath(str(record_path))))
    if len(paths) != len(set(paths)):
        raise SplitAuditError("Split source paths must differ")
    names = list(splits)
    for index, name in enumerate(names):
        left = splits[name]["manifest"]
        for other in names[index + 1:]:
            right = splits[other]["manifest"]
            if set(left["ids"]) & set(right["ids"]):
                raise SplitAuditError(f"Utterance overlap: {name}/{other}")
            if set(left["speakers"]) & set(right["speakers"]):
                raise SplitAuditError(f"Speaker overlap: {name}/{other}")
    return {"source_paths_distinct": True, "utterance_ids_disjoint": True,
            "speakers_disjoint": True}


def load_verified_split_report(report_path, expected_sha256, requested_splits=("train", "valid"),
                                repository_root=REPOSITORY_ROOT, expected_counts=EXPECTED_COUNTS):
    """Open only requested split files; verify test isolation using frozen metadata."""
    path = repo_path(report_path, repository_root)
    if not expected_sha256 or sha256_file(path) != expected_sha256:
        raise SplitAuditError("Split-audit report SHA-256 does not match the frozen configuration")
    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("schema_version") != 1 or report.get("sample_rate") != 16000 or report.get("channels") != 1:
        raise SplitAuditError("Unsupported split-audit schema or audio properties")
    validate_split_audits(report["splits"], expected_counts)
    for name in requested_splits:
        if name not in report["splits"]:
            raise SplitAuditError(f"Unknown requested split: {name}")
        frozen = report["splits"][name]
        manifest = audit_manifest(frozen["manifest"]["path"], expected_counts[name],
                                  frozen["manifest"]["speakers"], repository_root)
        if manifest != frozen["manifest"]:
            raise SplitAuditError(f"Manifest changed after preflight: {name}")
        cache = audit_hdf5_cache(frozen["cache"]["path"], manifest,
                                report["sample_rate"], report["channels"], repository_root)
        if cache != frozen["cache"]:
            raise SplitAuditError(f"HDF5 changed after preflight: {name}")
    return report
