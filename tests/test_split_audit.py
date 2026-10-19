"""Synthetic-data regression coverage for provenance and non-destructive prep."""
import copy
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import soundfile as sf

from dpsnn.data import split_audit as audit
from dpsnn.data.hdf5_prepare import create_hdf5
from dpsnn.data.voicebank_prepare import TRAIN_SPEAKERS
from tools.prepare_camera_ready_data import prepare_data


def make_manifest(root, name, speaker, sample_rate=16000):
    folder = root / name
    (folder / "clean").mkdir(parents=True)
    (folder / "noisy").mkdir()
    utterance_id = f"{speaker}_001"
    audio = np.linspace(-0.4, 0.4, 512, dtype=np.float32)
    paths = []
    for kind in ("noisy", "clean"):
        path = folder / kind / f"{utterance_id}.wav"
        sf.write(path, audio, sample_rate, subtype="FLOAT")
        paths.append(path)
    path = folder / f"{name}.csv"
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "duration", "noisy_wav", "clean_wav"])
        writer.writerow([utterance_id, 512/sample_rate, *map(str, paths)])
    return path


def make_report(root):
    splits = {}
    for name, speaker in (("train", "p227"), ("valid", "p226"), ("test", "p232")):
        csv_path = make_manifest(root, name, speaker)
        manifest = audit.audit_manifest(csv_path, 1, {speaker}, root)
        cache = audit.ensure_hdf5_cache(csv_path, csv_path.with_suffix(".hdf5"), manifest,
                                       repository_root=root)
        splits[name] = {"manifest": manifest, "cache": cache}
    report = {"schema_version": 1, "sample_rate": 16000, "channels": 1, "splits": splits}
    path = root / "audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path, report


@pytest.mark.parametrize("missing", ["train", "valid", "test"])
def test_each_missing_cache_is_created_without_changing_existing(tmp_path, missing):
    _, report = make_report(tmp_path)
    paths = {name: tmp_path / value["cache"]["path"] for name, value in report["splits"].items()}
    before = {name: audit.sha256_file(path) for name, path in paths.items()}
    paths[missing].unlink()
    calls = []

    def tracked_create(csv_path, hdf5_path, sample_rate, channels):
        calls.append(Path(csv_path).stem)
        create_hdf5(csv_path, hdf5_path, sample_rate, channels)

    for name, split in report["splits"].items():
        audit.ensure_hdf5_cache(split["manifest"]["path"], paths[name], split["manifest"],
                                create_fn=tracked_create, repository_root=tmp_path)
    assert calls == [missing]
    for name in set(paths) - {missing}:
        assert audit.sha256_file(paths[name]) == before[name]
    assert not list(tmp_path.rglob("*.partial"))


def test_partial_existing_cache_fails_without_overwrite(tmp_path):
    csv_path = make_manifest(tmp_path, "train", "p227")
    manifest = audit.audit_manifest(csv_path, 1, repository_root=tmp_path)
    cache_path = csv_path.with_suffix(".hdf5")
    cache_path.write_bytes(b"partial historical evidence")
    before = cache_path.read_bytes()
    with pytest.raises(audit.SplitAuditError, match="partial cache"):
        audit.ensure_hdf5_cache(csv_path, cache_path, manifest, repository_root=tmp_path)
    assert cache_path.read_bytes() == before


def test_destination_collision_preserves_other_writer(tmp_path):
    csv_path = make_manifest(tmp_path, "train", "p227")
    manifest = audit.audit_manifest(csv_path, 1, repository_root=tmp_path)
    cache_path = csv_path.with_suffix(".hdf5")

    def competing_create(csv_path, temporary_path, sample_rate, channels):
        create_hdf5(csv_path, temporary_path, sample_rate, channels)
        cache_path.write_bytes(b"another writer")

    with pytest.raises(FileExistsError):
        audit.ensure_hdf5_cache(csv_path, cache_path, manifest,
                                create_fn=competing_create, repository_root=tmp_path)
    assert cache_path.read_bytes() == b"another writer"
    assert not list(tmp_path.rglob("*.partial"))


@pytest.mark.parametrize("defect", ["id", "rate", "length", "nan"])
def test_invalid_existing_cache_is_rejected(tmp_path, defect):
    csv_path = make_manifest(tmp_path, "train", "p227")
    manifest = audit.audit_manifest(csv_path, 1, repository_root=tmp_path)
    cache_path = csv_path.with_suffix(".hdf5")
    audit.ensure_hdf5_cache(csv_path, cache_path, manifest, repository_root=tmp_path)
    with h5py.File(cache_path, "a") as cache:
        if defect == "id":
            cache["0"].attrs["ID"] = "p999_001"
        elif defect == "rate":
            cache.attrs["sr"] = 48000
        elif defect == "length":
            cache["0"].attrs["length"] = 100
        else:
            cache["0"]["clean"][0, 0] = np.nan
    before = audit.sha256_file(cache_path)
    with pytest.raises(audit.SplitAuditError):
        audit.ensure_hdf5_cache(csv_path, cache_path, manifest, repository_root=tmp_path)
    assert audit.sha256_file(cache_path) == before


@pytest.mark.parametrize("defect", ["missing_clean", "duplicate", "wrong_rate", "wrong_count"])
def test_manifest_pair_count_and_rate_failures(tmp_path, defect):
    path = make_manifest(tmp_path, "train", "p227", sample_rate=48000 if defect == "wrong_rate" else 16000)
    if defect == "missing_clean":
        (path.parent / "clean" / "p227_001.wav").unlink()
    elif defect == "duplicate":
        with path.open("a", encoding="utf-8") as stream:
            stream.write(path.read_text(encoding="utf-8").splitlines()[-1] + "\n")
    with pytest.raises(audit.SplitAuditError):
        audit.audit_manifest(path, 2 if defect == "wrong_count" else 1, repository_root=tmp_path)


@pytest.mark.parametrize("defect", ["same_path", "id_overlap", "speaker_overlap", "count"])
def test_frozen_split_disjointness(tmp_path, defect):
    _, report = make_report(tmp_path)
    splits = copy.deepcopy(report["splits"])
    if defect == "same_path":
        splits["valid"]["cache"]["path"] = splits["test"]["cache"]["path"]
    elif defect in {"id_overlap", "speaker_overlap"}:
        new_id = "p227_001" if defect == "id_overlap" else "p227_002"
        for record in splits["valid"].values():
            record["ids"] = [new_id]
            record["speakers"] = ["p227"]
    else:
        splits["valid"]["manifest"]["count"] = 2
    with pytest.raises(audit.SplitAuditError):
        audit.validate_split_audits(splits, {name: 1 for name in splits})


def test_train_validation_verification_does_not_access_test_files(tmp_path, monkeypatch):
    path, report = make_report(tmp_path)
    report_hash = audit.sha256_file(path)
    forbidden = tmp_path / "test"
    original_open, original_stat = Path.open, Path.stat
    original_hdf5_open = h5py.File

    def guard(path_value):
        if Path(path_value).is_relative_to(forbidden):
            raise AssertionError(f"Official-test file accessed: {path_value}")

    def guarded_open(self, *args, **kwargs):
        guard(self)
        return original_open(self, *args, **kwargs)

    def guarded_stat(self, *args, **kwargs):
        guard(self)
        return original_stat(self, *args, **kwargs)

    def guarded_hdf5(path_value, *args, **kwargs):
        guard(path_value)
        return original_hdf5_open(path_value, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(h5py, "File", guarded_hdf5)
    assert audit.load_verified_split_report(path, report_hash, repository_root=tmp_path,
                                             expected_counts={name: 1 for name in report["splits"]}) == report


def test_report_and_cache_hash_changes_are_rejected(tmp_path):
    path, report = make_report(tmp_path)
    report_hash = audit.sha256_file(path)
    kwargs = {"repository_root": tmp_path, "expected_counts": {name: 1 for name in report["splits"]}}
    with pytest.raises(audit.SplitAuditError, match="SHA-256"):
        audit.load_verified_split_report(path, "0" * 64, **kwargs)
    with h5py.File(tmp_path / report["splits"]["train"]["cache"]["path"], "a") as cache:
        cache["0"]["noisy"][0, 0] += 0.1
    with pytest.raises(audit.SplitAuditError, match="HDF5 changed"):
        audit.load_verified_split_report(path, report_hash, **kwargs)


def test_complete_preparation_supports_float_wav_and_preserves_sources(tmp_path):
    counts = {"train": 26, "valid": 2, "test": 2}
    source_hashes = {}
    for root_name, speakers, suffix in (("train", TRAIN_SPEAKERS, "trainset_28spk_wav"),
                                        ("test", ["p232", "p257"], "testset_wav")):
        for kind in ("clean", "noisy"):
            folder = tmp_path / root_name / f"{kind}_{suffix}"
            folder.mkdir(parents=True)
            for speaker in speakers:
                path = folder / f"{speaker}_001.wav"
                # Non-divisible frame count checks actual resampled duration.
                sf.write(path, np.linspace(-0.4, 0.4, 4801, dtype=np.float32), 48000, subtype="PCM_16")
                source_hashes[path] = audit.sha256_file(path)
    output = tmp_path / "corrected"
    report = prepare_data(tmp_path / "train", tmp_path / "test", output,
                          repository_root=tmp_path, expected_counts=counts)
    assert {name: split["manifest"]["count"] for name, split in report["splits"].items()} == counts
    assert set(report["splits"]["valid"]["manifest"]["speakers"]) == {"p226", "p287"}
    assert report["splits"]["test"]["manifest"]["frames"] == [1601, 1601]
    assert all(audit.sha256_file(path) == digest for path, digest in source_hashes.items())
    report_path = output / "split_audit.json"
    audit.load_verified_split_report(report_path, audit.sha256_file(report_path),
                                      repository_root=tmp_path, expected_counts=counts)
    before = audit.sha256_file(report_path)
    with pytest.raises(FileExistsError):
        prepare_data(tmp_path / "train", tmp_path / "test", output, repository_root=tmp_path,
                      expected_counts=counts)
    assert audit.sha256_file(report_path) == before
