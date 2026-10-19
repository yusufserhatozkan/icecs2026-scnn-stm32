"""Run records and checkpoint safeguards for corrected VoiceBank training."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from dpsnn.data.split_audit import (
    EXPECTED_COUNTS, REPOSITORY_ROOT, SplitAuditError,
    load_verified_split_report, relative_path, repo_path, sha256_file,
)


def write_json_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_training_inputs(args, config, repository_root=REPOSITORY_ROOT,
                           expected_counts=EXPECTED_COUNTS):
    """Resolve only requested data files; compare other paths lexically."""
    if args.load_ckpt_path:
        raise SplitAuditError("Corrected runs must start from scratch; checkpoint resume requires a separate audited procedure")
    if args.random_seeds:
        raise SplitAuditError("Corrected protocol requires seed 2020")
    if args.augment:
        raise SplitAuditError("Augmentation is disabled in the corrected protocol")
    if config.checkpoint.monitor != "val_loss" or config.checkpoint.get("mode", "min") != "min":
        raise SplitAuditError("Checkpoint selection must minimize val_loss")
    if config.checkpoint.save_top_k != 3 or config.checkpoint.save_last is not True:
        raise SplitAuditError("Corrected protocol saves top three checkpoints and last")
    if config.trainer.get("callbacks"):
        raise SplitAuditError("Additional configured callbacks, including early stopping, are not allowed")
    if config.sample_rate != 16000:
        raise SplitAuditError("Corrected protocol requires 16 kHz")
    if (args.L != 80 or args.stride != 40 or args.X != 1 or args.norm_type != 'lnorm'
            or not args.random_hops or config.batch_size != 64
            or config.frame_dur != 1.0 or config.context_dur != 0.01 or config.delay_dur != 0
            or config.optim.lr != 0.01 or config.trainer.precision != 'bf16-mixed'
            or config.trainer.accelerator != 'gpu' or config.trainer.gradient_clip_val != 1.0):
        raise SplitAuditError('Corrected model, batch, precision or optimizer settings differ from the frozen protocol')
    requested = (["test"] if args.test_ckpt_path else ["train", "valid"])
    if not args.test_ckpt_path and not args.skip_test_after_fit:
        requested.append("test")
    report_path = args.split_audit_report or config.get("split_audit_report")
    report_hash = args.split_audit_sha256 or config.get("split_audit_sha256")
    if not report_path:
        raise SplitAuditError("A frozen split_audit_report is required; run the separate preparation command first")
    report = load_verified_split_report(report_path, report_hash, requested,
                                        repository_root, expected_counts)
    root = Path(repository_root).resolve()
    for name in ("train", "valid", "test"):
        for prefix, record_name in (("csv", "manifest"), ("hdf5", "cache")):
            key = f"{prefix}_{name}"
            value = str(config[key])
            # This comparison deliberately does not stat/resolve a test path.
            configured = os.path.normcase(os.path.abspath(os.path.join(root, value)))
            frozen = os.path.normcase(os.path.abspath(os.path.join(root, report["splits"][name][record_name]["path"])))
            if configured != frozen:
                raise SplitAuditError(f"Configured {key} differs from the frozen split report")
            if name in requested:
                config[key] = str(repo_path(value, root))
    config.split_audit_report = str(repo_path(report_path, root))
    config.split_audit_sha256 = report_hash
    for name in requested:
        record = report["splits"][name]["cache"]
        print(f"Verified {name}: {record['path']}; utterances={record['count']}; "
              f"speakers={','.join(record['speakers'])}; sha256={record['sha256']}")
    return report


def reserve_run_directory(args, config, repository_root=REPOSITORY_ROOT):
    value = args.run_dir or config.get("run_dir")
    if not value:
        raise SplitAuditError("--run_dir must name a new repository directory")
    run_dir = repo_path(value, repository_root)
    run_dir.mkdir(parents=True, exist_ok=False)
    config.run_dir = str(run_dir)
    config.trainer.default_root_dir = str(run_dir)
    # Trainer callback audio (when explicitly requested) also stays in the run.
    config.output_folder = str(run_dir)
    return run_dir


def checkpoint_callback(config, run_dir):
    return ModelCheckpoint(dirpath=str(Path(run_dir) / "checkpoints"), monitor="val_loss",
                           mode="min", save_top_k=3, save_last=True,
                           filename=config.checkpoint.filename)


def validate_model_parameters(model, args):
    if not args.scnn_only or args.N != args.B or args.B != args.H or args.N not in (64, 128):
        raise SplitAuditError("Corrected models require SCNN-only N=B=H of 64 or 128")
    if args.no_pointwise:
        raise SplitAuditError("The conference baselines retain their pointwise layers")
    expected = 71299 if args.N == 128 else 23363
    actual = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if actual != expected:
        raise SplitAuditError(f"Trainable parameter count {actual} != expected {expected}")
    return actual


def record_resolved_run(run_dir, args, config, split_report, model, train_loader=None, valid_loader=None,
                        repository_root=REPOSITORY_ROOT):
    resolved = {"arguments": vars(args), "configuration": OmegaConf.to_container(config, resolve=True)}
    serialized = json.dumps(resolved, sort_keys=True, separators=(",", ":"), allow_nan=False)
    write_json_exclusive(Path(run_dir) / "resolved_config.json", resolved)
    info = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_directory": relative_path(run_dir, repository_root),
            "resolved_config_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "split_audit_sha256": config.split_audit_sha256,
            "seed": 2020, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "input_samples": int(model.hparams["input_dim"]),
            "context_samples": int(model.hparams["context_dim"]),
            "output_samples": (model.time_steps - 1) * model.stride + model.L,
            "time_steps": model.time_steps, "batch_size": config.batch_size,
            "full_training_batches_per_epoch": len(train_loader) if train_loader is not None else None,
            "full_validation_batches_per_epoch": len(valid_loader) if valid_loader is not None else None,
            "split_counts": {name: value["manifest"]["count"] for name, value in split_report["splits"].items()},
            "test_after_fit": not args.skip_test_after_fit,
            "checkpoint": {"monitor": "val_loss", "mode": "min", "save_top_k": 3, "save_last": True},
            "loss": "100 + negative_si_sdr + 0.001 * mse",
            "optimizer": {"name": "Adam", "initial_lr": config.optim.lr,
                          "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0},
            "scheduler": {"name": "ExponentialLR", "interval": "step",
                          "gamma_rule": "max(0.9999, (0.001 / initial_lr) ** (1 / estimated_stepping_batches))"}}
    def git(*arguments):
        return subprocess.check_output(['git', *arguments], cwd=repository_root).decode('utf-8').strip()
    info['git_commit'] = git('rev-parse', 'HEAD')
    info['git_status'] = git('status', '--short')
    info['worktree_clean'] = not info['git_status']
    info['command'] = [os.sys.executable, *os.sys.argv]
    info['requirements_sha256'] = sha256_file(Path(repository_root) / 'requirements.txt')
    if info['git_status']:
        with (Path(run_dir) / 'tracked_source.diff').open('xb') as stream:
            stream.write(subprocess.check_output(['git', 'diff', 'HEAD', '--binary'], cwd=repository_root))
    write_json_exclusive(Path(run_dir) / "run_manifest.json", info)
    return info


class ProtocolLogger(pl.Callback):
    """Record finite losses, epoch progress and checkpoint selection."""
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.best_val_loss = float("inf")
        self.train_losses, self.validation_losses = [], []
        self.train_batch_seconds, self.validation_batch_seconds = [], []
        self.epochs = []
        self.runtime = {}

    def state_dict(self):
        return {"best_val_loss": self.best_val_loss}

    def load_state_dict(self, state_dict):
        self.best_val_loss = state_dict["best_val_loss"]

    def on_train_start(self, trainer, pl_module):
        self.started = time.perf_counter()
        scheduler = trainer.lr_scheduler_configs[0].scheduler
        self.runtime = {"device": str(pl_module.device), "precision": str(trainer.precision),
                        "estimated_stepping_batches": int(trainer.estimated_stepping_batches),
                        "training_batches_per_epoch": int(trainer.num_training_batches),
                        "validation_batches_per_epoch": list(trainer.num_val_batches),
                        "scheduler_gamma": scheduler.gamma,
                        "optimizer_groups": [{key: value for key, value in group.items() if key != "params"}
                                             for group in trainer.optimizers[0].param_groups]}
        if pl_module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pl_module.device)
            self.runtime["gpu_name"] = torch.cuda.get_device_name(pl_module.device)
        with (self.run_dir / "epoch_log.md").open("x", encoding="utf-8") as stream:
            stream.write("| Epoch | val_loss (minimize) | negative SI-SDR loss (minimize) | Notes |\n"
                         "|---|---|---|---|\n")
        write_json_exclusive(self.run_dir / 'runtime_start.json', self.runtime)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.train_batch_started = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        value = outputs["loss"] if isinstance(outputs, dict) else outputs
        loss = float(value.detach())
        if not math.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at epoch {trainer.current_epoch}, batch {batch_idx}")
        self.train_losses.append(loss)
        self.train_batch_seconds.append(time.perf_counter() - self.train_batch_started)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.validation_batch_started = time.perf_counter()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = float(outputs.detach())
        if not math.isfinite(loss):
            raise FloatingPointError(f"Nonfinite validation loss at epoch {trainer.current_epoch}, batch {batch_idx}")
        if not trainer.sanity_checking:
            self.validation_losses.append(loss)
            self.validation_batch_seconds.append(time.perf_counter() - self.validation_batch_started)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        val_loss, negative_sisdr = float(metrics["val_loss"]), float(metrics["val_sisnr"])
        if not math.isfinite(val_loss) or not math.isfinite(negative_sisdr):
            raise FloatingPointError("Nonfinite validation epoch metric")
        is_best = val_loss < self.best_val_loss
        self.best_val_loss = min(self.best_val_loss, val_loss)
        self.epochs.append({"epoch": trainer.current_epoch, "val_loss": val_loss,
                            "negative_si_sdr_loss": negative_sisdr, "new_minimum_val_loss": is_best})
        progress = {**self.epochs[-1], 'utc': datetime.now(timezone.utc).isoformat(),
                    'global_step': trainer.global_step,
                    'elapsed_seconds': time.perf_counter() - self.started}
        with (self.run_dir / 'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(progress, allow_nan=False) + '\n')
        print('EPOCH_PROGRESS ' + json.dumps(progress, allow_nan=False), flush=True)
        with (self.run_dir / "epoch_log.md").open("a", encoding="utf-8") as stream:
            note = "new minimum val_loss" if is_best else ""
            stream.write(f"| {trainer.current_epoch} | {val_loss:.8f} | {negative_sisdr:.8f} | {note} |\n")

    def on_fit_end(self, trainer, pl_module):
        def timing(values):
            return {"count": len(values), "total_seconds": sum(values),
                    "mean_seconds": statistics.mean(values) if values else None,
                    "median_seconds": statistics.median(values) if values else None,
                    "seconds": values}
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
            self.runtime["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(pl_module.device)
            self.runtime["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(pl_module.device)
        self.runtime.update(fit_seconds=time.perf_counter() - self.started,
                            train_batch_timing=timing(self.train_batch_seconds),
                            validation_batch_timing=timing(self.validation_batch_seconds),
                            train_losses=self.train_losses, validation_losses=self.validation_losses,
                            epochs=self.epochs, all_recorded_losses_finite=True)
        write_json_exclusive(self.run_dir / "fit_evidence.json", self.runtime)


def record_checkpoint_selection(run_dir, callback, repository_root=REPOSITORY_ROOT):
    """Read the framework-selected checkpoint after fitting has fully completed."""
    if not callback.best_model_path or callback.best_model_score is None:
        raise SplitAuditError("Training finished without a selected validation checkpoint")
    best_path = repo_path(callback.best_model_path, repository_root)
    score = float(callback.best_model_score)
    if not math.isfinite(score):
        raise SplitAuditError("Nonfinite selected validation loss")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    record = {"monitor": callback.monitor, "mode": callback.mode,
              "best_model_path": relative_path(best_path, repository_root),
              "best_model_score": score, "best_epoch": int(checkpoint["epoch"]),
              "best_checkpoint_sha256": sha256_file(best_path),
              "last_model_path": relative_path(callback.last_model_path, repository_root),
              "last_checkpoint_sha256": sha256_file(callback.last_model_path)}
    write_json_exclusive(Path(run_dir) / "checkpoint_selection.json", record)
    print("Framework checkpoint selection: " + json.dumps(record))
    return record
