"""Prepare a new, audited VoiceBank-DEMAND cache without changing old data.

Run from the repository root with ``python -m tools.prepare_camera_ready_data``.
This command prepares data only; it never trains a model or computes quality
metrics. An existing output directory is always rejected.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import wave

import h5py
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm

from dpsnn.data.split_audit import (
    EXPECTED_COUNTS, REPOSITORY_ROOT, VALID_SPEAKERS, TEST_SPEAKERS,
    SplitAuditError, audit_manifest, ensure_hdf5_cache, file_record,
    relative_path, repo_path, sha256_file, validate_split_audits,
)
from dpsnn.data.voicebank_prepare import TRAIN_SPEAKERS


CSV_HEADER = ["ID", "duration", "noisy_wav", "noisy_wav_format", "noisy_wav_opts",
              "clean_wav", "clean_wav_format", "clean_wav_opts",
              "char", "char_format", "char_opts"]


def _wav_record(path, repository_root):
    with wave.open(str(path), "rb") as stream:
        audio = {"sample_rate": stream.getframerate(), "channels": stream.getnchannels(),
                 "frames": stream.getnframes(), "sample_width_bytes": stream.getsampwidth()}
    if audio["sample_rate"] != 48000 or audio["channels"] != 1 or audio["frames"] <= 0:
        raise SplitAuditError(f"Expected nonempty mono 48 kHz source: {path}")
    return {**file_record(path, repository_root), **audio}


def inspect_raw_pairs(train_root, test_root, repository_root=REPOSITORY_ROOT,
                      expected_counts=EXPECTED_COUNTS):
    """Audit every source file before reserving any output directory."""
    roots = {"development": repo_path(train_root, repository_root),
             "test": repo_path(test_root, repository_root)}
    suffixes = {"development": "trainset_28spk_wav", "test": "testset_wav"}
    pairs = {"train": [], "valid": [], "test": []}
    inventory = []
    for source_name, root in roots.items():
        clean_dir = root / ("clean_" + suffixes[source_name])
        noisy_dir = root / ("noisy_" + suffixes[source_name])
        maps = []
        for folder in (clean_dir, noisy_dir):
            if not folder.is_dir():
                raise SplitAuditError(f"Source directory missing: {folder}")
            wavs = sorted(folder.glob("*.wav"))
            mapping = {path.stem: path for path in wavs}
            if len(mapping) != len(wavs) or len({key.casefold() for key in mapping}) != len(mapping):
                raise SplitAuditError(f"Duplicate source IDs: {folder}")
            maps.append(mapping)
        clean_files, noisy_files = maps
        if set(clean_files) != set(noisy_files):
            raise SplitAuditError(f"Unmatched source pairs in {source_name}: "
                                  f"clean-only={sorted(set(clean_files)-set(noisy_files))}, "
                                  f"noisy-only={sorted(set(noisy_files)-set(clean_files))}")
        expected = (expected_counts["train"] + expected_counts["valid"]
                    if source_name == "development" else expected_counts["test"])
        if len(clean_files) != expected:
            raise SplitAuditError(f"Raw {source_name} pair count {len(clean_files)} != {expected}")
        source_speakers = {key.split("_", 1)[0] for key in clean_files}
        expected_speakers = set(TRAIN_SPEAKERS) if source_name == "development" else TEST_SPEAKERS
        if source_speakers != expected_speakers:
            raise SplitAuditError(f"Unexpected raw {source_name} speakers: {sorted(source_speakers)}")
        inventory.append({"source": source_name,
                          "clean_directory": relative_path(clean_dir, repository_root),
                          "noisy_directory": relative_path(noisy_dir, repository_root),
                          "clean_files": len(clean_files), "noisy_files": len(noisy_files),
                          "matched_pairs": len(clean_files), "unmatched_files": 0,
                          "duplicate_ids": 0, "speakers": sorted(source_speakers)})
        for utterance_id in tqdm(sorted(clean_files), desc=f"Audit {source_name}"):
            clean = _wav_record(clean_files[utterance_id], repository_root)
            noisy = _wav_record(noisy_files[utterance_id], repository_root)
            if clean["frames"] != noisy["frames"]:
                raise SplitAuditError(f"Raw clean/noisy length mismatch: {utterance_id}")
            speaker = utterance_id.split("_", 1)[0]
            split = "test" if source_name == "test" else ("valid" if speaker in VALID_SPEAKERS else "train")
            pairs[split].append({"id": utterance_id, "speaker": speaker,
                                 "source": {"clean": clean, "noisy": noisy}})
    for name, records in pairs.items():
        if len(records) != expected_counts[name]:
            raise SplitAuditError(f"Incorrect derived {name} count: {len(records)}")
    return pairs, inventory


def _compare_historical(old_path, new_path):
    """Compare all samples by utterance ID; no inference or quality metrics."""
    result = {"old_count": 0, "new_count": 0, "matched_ids": 0,
              "missing_old_ids": [], "extra_old_ids": [], "mismatching_ids": [],
              "identical_pairs": 0, "maximum_absolute_difference": 0.0}
    with h5py.File(old_path, "r") as old, h5py.File(new_path, "r") as new:
        def by_id(cache):
            mapping = {}
            for group_name in cache:
                value = cache[group_name].attrs["ID"]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                if str(value) in mapping:
                    raise SplitAuditError(f"Historical comparison found duplicate ID {value}")
                mapping[str(value)] = group_name
            return mapping
        old_ids, new_ids = by_id(old), by_id(new)
        result["old_count"], result["new_count"] = len(old_ids), len(new_ids)
        result["missing_old_ids"] = sorted(set(new_ids) - set(old_ids))
        result["extra_old_ids"] = sorted(set(old_ids) - set(new_ids))
        for utterance_id in tqdm(sorted(set(old_ids) & set(new_ids)), desc="Historical comparison"):
            result["matched_ids"] += 1
            identical = True
            for name in ("clean", "noisy"):
                before = old[old_ids[utterance_id]][name][()]
                after = new[new_ids[utterance_id]][name][()]
                if before.shape != after.shape:
                    identical = False
                    continue
                if not np.isfinite(before).all() or not np.isfinite(after).all():
                    raise SplitAuditError(f"Nonfinite cache audio: {utterance_id}/{name}")
                difference = float(np.max(np.abs(before - after)))
                result["maximum_absolute_difference"] = max(result["maximum_absolute_difference"], difference)
                identical = identical and np.array_equal(before, after)
            if identical:
                result["identical_pairs"] += 1
            else:
                result["mismatching_ids"].append(utterance_id)
    return result


def prepare_data(train_root, test_root, output_root, historical_cache_root=None,
                 repository_root=REPOSITORY_ROOT, expected_counts=EXPECTED_COUNTS):
    output_root = repo_path(output_root, repository_root)
    if output_root.exists():
        raise FileExistsError(f"Refusing existing output directory: {output_root}")
    pairs, inventory = inspect_raw_pairs(train_root, test_root, repository_root, expected_counts)
    historical = {}
    old_root = repo_path(historical_cache_root, repository_root) if historical_cache_root else None
    if old_root is not None:
        for split in expected_counts:
            for suffix in ("csv", "hdf5"):
                path = old_root / f"{split}.{suffix}"
                if path.is_file():
                    record = file_record(path, repository_root)
                    record["mtime_utc"] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
                    historical[f"{split}.{suffix}"] = record
    # All source validation and historical-file inventory above are read-only.
    output_root.mkdir(parents=True, exist_ok=False)
    audio_root, save_root = output_root / "audio_16k", output_root / "save"
    save_root.mkdir()
    resampler = Resample(orig_freq=48000, new_freq=16000)
    splits = {}
    for name, records in pairs.items():
        suffix = "testset_wav_16k" if name == "test" else "trainset_28spk_wav_16k"
        for kind in ("clean", "noisy"):
            (audio_root / f"{kind}_{suffix}").mkdir(parents=True, exist_ok=True)
        rows = []
        for record in tqdm(records, desc=f"Resample {name}"):
            resampled = {}
            for kind in ("clean", "noisy"):
                source = repo_path(record["source"][kind]["path"], repository_root)
                if sha256_file(source) != record["source"][kind]["sha256"]:
                    raise SplitAuditError(f"Source content changed after audit: {source}")
                signal, sample_rate = torchaudio.load(str(source))
                if sample_rate != 48000 or signal.shape != (1, record["source"][kind]["frames"]):
                    raise SplitAuditError(f"Source audio changed after audit: {source}")
                signal = resampler(signal)
                if not torch.isfinite(signal).all():
                    raise SplitAuditError(f"Nonfinite resampled waveform: {record['id']}/{kind}")
                destination = audio_root / f"{kind}_{suffix}" / f"{record['id']}.wav"
                if destination.exists():
                    raise FileExistsError(destination)
                # Preserve the existing preparation helper's default WAV encoding.
                torchaudio.save(str(destination), signal, sample_rate=16000)
                audio_info = torchaudio.info(str(destination))
                info = {"sample_rate": audio_info.sample_rate, "channels": audio_info.num_channels,
                        "frames": audio_info.num_frames, "bits_per_sample": audio_info.bits_per_sample,
                        "encoding": audio_info.encoding}
                if info["sample_rate"] != 16000 or info["channels"] != 1:
                    raise SplitAuditError(f"Unexpected resampled WAV properties: {destination}")
                resampled[kind] = {**file_record(destination, repository_root), **info}
            if resampled["clean"]["frames"] != resampled["noisy"]["frames"]:
                raise SplitAuditError(f"Resampled pair length mismatch: {record['id']}")
            record["resampled"] = resampled
            noisy_path = repo_path(resampled["noisy"]["path"], repository_root)
            clean_path = repo_path(resampled["clean"]["path"], repository_root)
            rows.append([record["id"], f"{resampled['noisy']['frames']/16000:.6f}",
                         str(noisy_path), "wav", "", str(clean_path), "wav", "", "", "string", ""])
        csv_path = save_root / f"{name}.csv"
        with csv_path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(CSV_HEADER)
            writer.writerows(rows)
        manifest = audit_manifest(csv_path, expected_counts[name],
                                  {record["speaker"] for record in records}, repository_root)
        cache = ensure_hdf5_cache(csv_path, save_root / f"{name}.hdf5", manifest,
                                 repository_root=repository_root)
        splits[name] = {"manifest": manifest, "cache": cache, "pairs": records}
    overlap_checks = validate_split_audits(splits, expected_counts)
    comparisons = {}
    if old_root is not None:
        for name in expected_counts:
            old_path = old_root / f"{name}.hdf5"
            if old_path.is_file():
                comparisons[name] = _compare_historical(old_path, save_root / f"{name}.hdf5")
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "sample_rate": 16000, "channels": 1,
              "preparation": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "torchaudio": str(torchaudio.__version__),
                              "resampler": "torchaudio.transforms.Resample",
                              "orig_freq": 48000, "new_freq": 16000,
                              "lowpass_filter_width": resampler.lowpass_filter_width,
                              "rolloff": resampler.rolloff,
                              "resampling_method": resampler.resampling_method,
                              "beta": resampler.beta, "torch_threads": torch.get_num_threads(),
                              "wav_save_encoding": "torchaudio.save default"},
              "raw_inventory": inventory, "splits": splits, "overlap_checks": overlap_checks,
              "historical_files": historical, "historical_tensor_comparison": comparisons}
    report_path = output_root / "split_audit.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": file_record(report_path, repository_root),
                      "counts": {name: split["manifest"]["count"] for name, split in splits.items()},
                      "historical_identical_pairs": {name: value["identical_pairs"]
                                                       for name, value in comparisons.items()}}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--historical-cache-root")
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="CPU threads for preparation only; recorded in the audit (default: 1)")
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    prepare_data(args.train_root, args.test_root, args.output_root, args.historical_cache_root)


if __name__ == "__main__":
    main()
