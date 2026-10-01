import time
import argparse, os, sys
import random

from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
import torchaudio

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


from dpsnn.data.wave_dataset2 import worker_init_fn
from dpsnn.data.wave_dataset2 import ContextSepDataset, EvaluationDataset
from dpsnn.data.augment import remix
from dpsnn.data.hdf5_prepare import create_hdf5
from dpsnn.data.voicebank_prepare import prepare_voicebank, download_vctk
from dpsnn.layers.sdr import singlesrc_neg_sisdr

from dpsnn.models.dp_binary_net import StreamSpikeNet
from dpsnn.training_protocol import (
    ProtocolLogger, checkpoint_callback, record_checkpoint_selection,
    record_resolved_run, reserve_run_directory, validate_model_parameters,
    verify_training_inputs,
)

def optimize_seeding():
    torch.backends.cudnn.benchmark = True
    _seed_ = 2020
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(_seed_)
    random.seed(_seed_)

def randomize_seeding():
    torch.seed()  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.backends.cudnn.deterministic = False
    np.random.seed()

parser = argparse.ArgumentParser()
parser.add_argument('--random_seeds', action='store_true')
parser.add_argument('--config', type=str, required=False, default="conference.yaml")
parser.add_argument('--test_ckpt_path', type=str, required=False)
parser.add_argument('--load_ckpt_path', type=str, required=False)
parser.add_argument('--frame_dur', type=float, required=False)
parser.add_argument('--context_dur', type=float, required=False)
parser.add_argument('--delay_dur', type=float, required=False)
parser.add_argument('-L', type=int, required=False, default=20)
parser.add_argument('--stride', type=int, required=False, default=10)
parser.add_argument('-N', type=int, required=False, default=256)
parser.add_argument('-H', type=int, required=False, default=256)
parser.add_argument('-B', type=int, required=False, default=256)
parser.add_argument('-K', type=int, required=False, default=100)
parser.add_argument('--alpha', type=float, required=False, default=0.5)
parser.add_argument('--beta', type=float, required=False, default=0.5)
parser.add_argument('--rho', type=float, default=0.0, help='Rho')
parser.add_argument('--lmbda', type=float, default=2.0, help='Lambda')
parser.add_argument('--liquid', action='store_true')
# parser.add_argument('-R', type=int, required=False, default=1)
parser.add_argument('-X', type=int, required=False, default=1)
parser.add_argument('--sr', type=int, required=False)
parser.add_argument('--bi_direction', action='store_true')
parser.add_argument('--augment', action='store_true')
parser.add_argument('--neuro_type', type=str, required=False, default="plif")
parser.add_argument('--batch_size', type=int, required=False)
parser.add_argument('--max_epochs', type=int, required=False)
parser.add_argument('--lr', type=float, required=False)
parser.add_argument('--precision', type=str, required=False, help="16, 32, bf16")
parser.add_argument('--devices', nargs='+', type=int, default=[0])
parser.add_argument('--device_num', type=int, required=False)
parser.add_argument('--random_hops', action='store_false')
parser.add_argument('--scnn_only', action='store_true', help='SCNN-only variant: skip SRNN in each block')
parser.add_argument('--norm_type', default='lnorm', choices=['lnorm'])
parser.set_defaults(no_pointwise=False)
parser.add_argument('--exp_name', type=str, default='conference',
                    help='Human-readable experiment tag recorded in the resolved configuration.')
parser.add_argument('--skip_test_after_fit', action='store_true',
                    help='Do not open test data or initialize/evaluate test callbacks during training.')
parser.add_argument('--split_audit_report', type=str,
                    help='Repository-relative frozen split-audit JSON; overrides the configuration.')
parser.add_argument('--split_audit_sha256', type=str,
                    help='Expected SHA-256 of the frozen split-audit JSON.')
parser.add_argument('--run_dir', type=str,
                    help='New repository-relative directory for all records, logs and checkpoints.')


def rank_print(info):
    rank_zero_info(info)


script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)
args = None
config = None


def configure_runtime(argv=None):
    """Parse only when explicitly invoked, so protocol helpers can be tested."""
    global args, config
    args = parser.parse_args(argv)
    rank_print(args)
    optimize_seeding()
    if args.random_seeds:
        randomize_seeding()
    conf_file_path = os.path.join(script_dir, args.config)
    config = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(conf_file_path), resolve=True))
    for argument, key in (("frame_dur", "frame_dur"), ("context_dur", "context_dur"),
                          ("delay_dur", "delay_dur"), ("sr", "sample_rate")):
        if getattr(args, argument) is not None:
            config[key] = getattr(args, argument)
    if args.device_num:
        config.trainer.devices = args.device_num
    elif args.devices:
        config.trainer.devices = args.devices
        args.device_num = len(args.devices)
    if args.precision:
        config.trainer.precision = int(args.precision) if args.precision.isdecimal() else args.precision
    if args.max_epochs:
        config.trainer.max_epochs = args.max_epochs
    if args.lr:
        config.optim.lr = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    config.batch_size //= args.device_num
    rank_print(f"Resolved runtime configuration:\n{OmegaConf.to_yaml(config)}")
    return args, config



def normalize_estimates(est, mix):
    """Normalizes estimates according to the mixture maximum amplitude

    Args:
        est_np (np.array): Estimates with shape (n_src, time).
        mix_np (np.array): One mixture with shape (time, ).

    """
    # mix_max = torch.max(torch.abs(mix))
    # return est * mix_max / torch.max(torch.abs(est))
    return est / est.abs().max(dim=1, keepdim=True)[0]


def start_func():
    if args.test_ckpt_path or not args.skip_test_after_fit:
        raise ValueError("Use --skip_test_after_fit; evaluation/eval_streaming.py handles testing")
    # Data preparation and evaluation are separate from training.
    # In train-only mode this verification opens train/validation files only.
    split_report = verify_training_inputs(args, config)
    run_dir = reserve_run_directory(args, config)

    start_context_dur = args.X * config.context_dur

    if args.test_ckpt_path is None:
        random_hops = False if args.augment else args.random_hops
        train_dataset = ContextSepDataset(
            hdf_file=config.hdf5_train,
            frame_dur=config.frame_dur,
            sr=config.sample_rate,
            channels=1,
            start_context_dur=start_context_dur,
            end_context_dur=config.delay_dur,
            random_hops=random_hops)
        def remix_collate(batch):
            inputs, targets = default_collate(batch)
            noisy, clean = inputs[1], targets[1]
            noisy_perm, clean = remix(noisy-clean, clean)
            inputs[1] = noisy_perm
            return inputs, targets

        collate_fn = remix_collate if args.augment else default_collate
        shuffle = False if args.augment else True
        train_dataloader = DataLoader(train_dataset,
                                batch_size=config.batch_size,
                                shuffle=shuffle,
                                num_workers=0,  # 0 required on Windows — h5py can't be pickled
                                collate_fn=collate_fn,
                                worker_init_fn=worker_init_fn)


        valid_dataset = ContextSepDataset(
            hdf_file=config.hdf5_valid,
            frame_dur=config.frame_dur,
            sr=config.sample_rate,
            channels=1,
            start_context_dur=start_context_dur,
            end_context_dur=config.delay_dur,
            random_hops=False)
        valid_dataloader = DataLoader(valid_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=0,  # 0 required on Windows — h5py can't be pickled
                                worker_init_fn=worker_init_fn)

    test_requested = args.test_ckpt_path is not None or not args.skip_test_after_fit
    if test_requested:
        test_dataset = EvaluationDataset(
            hdf_file=config.hdf5_test,
            frame_dur=config.frame_dur,
            sr=config.sample_rate,
            channels=1,
            start_context_dur=start_context_dur,
            end_context_dur=config.delay_dur)
        test_dataloader = DataLoader(test_dataset,
                                    batch_size=None,
                                    shuffle=False,
                                    num_workers=0)


    if args.test_ckpt_path is None:
        train_shapes = train_dataset.get_shapes()
    else:
        train_shapes = test_dataset.get_shapes()
    print(f"shapes: {train_shapes}")
    input_dim = train_shapes["input_size"]
    output_dim = train_shapes["output_size"]
    assert(train_shapes["start_context_size"] % args.X == 0)
    context_dim = train_shapes["start_context_size"] // args.X
    delay_dim = train_shapes["end_context_size"]
    rank_print(f"context_dim: {context_dim}")


    selection_callback = checkpoint_callback(config, run_dir)
    callbacks = [ProtocolLogger(run_dir), selection_callback]
    logger = CSVLogger(save_dir=str(run_dir), name='metrics', version=0)
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **config.trainer)
    assert not (args.load_ckpt_path and args.test_ckpt_path), \
        "Please make sure not to set load_ckpt_path and test_ckpt_path together!"

    spike_net = StreamSpikeNet(input_dim, context_dim,
                               sr=config.sample_rate,
                               L=args.L, stride=args.stride,
                               N=args.N, B=args.B, H=args.H, X=args.X,
                               learning_rate=config.optim.lr,
                               scnn_only=args.scnn_only,
                               norm_type=args.norm_type,
                               no_pointwise=args.no_pointwise)

    print(spike_net)
    validate_model_parameters(spike_net, args)
    record_resolved_run(run_dir, args, config, split_report, spike_net,
                        train_dataloader if args.test_ckpt_path is None else None,
                        valid_dataloader if args.test_ckpt_path is None else None)

    if args.test_ckpt_path is None:  # training
        if args.load_ckpt_path:
            load_ckpt_path = os.path.join(script_dir, args.load_ckpt_path)
            rank_print(f"loading model from \"{load_ckpt_path}\"")
        else:
            load_ckpt_path = None
            rank_print(f"training from scratch")

        start = time.time()
        rank_print(f"training starts at: {time.asctime(time.localtime(start))}")
        trainer.fit(spike_net, train_dataloader, valid_dataloader, ckpt_path=load_ckpt_path)
        end = time.time()
        rank_print(f"training ends at: {time.asctime(time.localtime(end))}, time elapsed {(end-start)/60:.2f} min")

        record_checkpoint_selection(run_dir, selection_callback)
        if test_requested:
            rank_print(f"\n\ntesting with best:")
            trainer.test(spike_net, ckpt_path="best", dataloaders=test_dataloader)
    else:  # testing
        test_ckpt_path = os.path.join(script_dir, args.test_ckpt_path)
        rank_print(f"test_ckpt_path: {test_ckpt_path}")

        trainer.test(spike_net, ckpt_path=test_ckpt_path, dataloaders=test_dataloader)


# guard in the main module to avoid creating subprocesses recursively.
# https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing
if __name__ == '__main__':
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    torch.set_float32_matmul_precision('high')
    configure_runtime()
    start_func()
