"""Critical train-only fitting and framework checkpoint-selection regressions."""
import json
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset

from dpsnn.models.dp_binary_net import StreamSpikeNet
from dpsnn.training_protocol import ProtocolLogger, validate_model_parameters
from egs.voicebank import vctk_trainer as entry


@pytest.mark.parametrize('width,no_pointwise,count', [(128, False, 71299), (64, False, 23363)])
def test_real_model_parameter_counts(width, no_pointwise, count):
    model = StreamSpikeNet(16160, 160, sr=16000, L=80, stride=40, N=width, B=width, H=width,
                           X=1, scnn_only=True, no_pointwise=no_pointwise)
    args = SimpleNamespace(scnn_only=True, N=width, B=width, H=width, no_pointwise=no_pointwise)
    assert validate_model_parameters(model, args) == count


class TinyData(Dataset):
    opened = []
    def __init__(self, hdf_file, **kwargs):
        assert hdf_file in ('TRAIN_ONLY', 'VALID_ONLY')
        self.opened.append(hdf_file)
    def __len__(self):
        return 2
    def __getitem__(self, index):
        return torch.ones(1)
    def get_shapes(self):
        return dict(input_size=400, output_size=240, start_context_size=160, end_context_size=0)


class TinyModel(pl.LightningModule):
    def __init__(self, input_dim, context_dim, **kwargs):
        super().__init__()
        self.save_hyperparameters('input_dim', 'context_dim')
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.time_steps, self.stride, self.L = 5, 40, 80
    def training_step(self, batch, batch_idx):
        return self.weight.square()
    def validation_step(self, batch, batch_idx):
        value = torch.tensor([4., 1., 3., 2., 5.][self.current_epoch])
        self.log('val_loss', value, on_epoch=True, batch_size=1)
        self.log('val_sisnr', -torch.tensor(float(self.current_epoch)), on_epoch=True, batch_size=1)
        return value
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return dict(optimizer=optimizer, lr_scheduler=dict(
            scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999), interval='step'))


def test_real_tiny_fit_never_constructs_test_and_keeps_true_top_three(tmp_path, monkeypatch):
    args = entry.parser.parse_args(['--skip_test_after_fit', '--run_dir', str(tmp_path / 'run'),
                                    '-L', '80', '--stride', '40', '-N', '64', '-B', '64', '-H', '64', '--scnn_only'])
    config = OmegaConf.create(dict(hdf5_train='TRAIN_ONLY', hdf5_valid='VALID_ONLY', hdf5_test='TEST_FORBIDDEN',
        sample_rate=16000, frame_dur=1., context_dur=.01, delay_dur=0., batch_size=1,
        split_audit_sha256='synthetic', optim=dict(lr=.01),
        trainer=dict(max_epochs=5, accelerator='cpu', devices=1, precision=32, enable_progress_bar=False,
                     enable_model_summary=False, num_sanity_val_steps=1, log_every_n_steps=1),
        checkpoint=dict(filename='{epoch}-{val_loss:.4f}')))
    report = {'splits': {name: {'manifest': {'count': 1}} for name in ('train', 'valid', 'test')}}
    monkeypatch.setattr(entry, 'args', args)
    monkeypatch.setattr(entry, 'config', config)
    monkeypatch.setattr(entry, 'verify_training_inputs', lambda *args: report)
    monkeypatch.setattr(entry, 'ContextSepDataset', TinyData)
    monkeypatch.setattr(entry, 'StreamSpikeNet', TinyModel)
    monkeypatch.setattr(entry, 'validate_model_parameters', lambda *args: 1)
    def forbidden(*args, **kwargs):
        pytest.fail('Test data or preparation or trainer.test reached during train-only fit')
    for name in ('EvaluationDataset', 'prepare_voicebank', 'download_vctk', 'create_hdf5'):
        monkeypatch.setattr(entry, name, forbidden)
    monkeypatch.setattr(pl.Trainer, 'test', forbidden)
    TinyData.opened = []
    entry.optimize_seeding()
    entry.start_func()
    assert TinyData.opened == ['TRAIN_ONLY', 'VALID_ONLY']
    run = tmp_path / 'run'
    selected = json.loads((run / 'checkpoint_selection.json').read_text())
    assert selected['best_epoch'] == 1 and selected['best_model_score'] == 1.
    checkpoints = list((run / 'checkpoints').glob('*.ckpt'))
    assert len(checkpoints) == 4  # Three selected files plus last.
    assert {torch.load(p, weights_only=True)['epoch'] for p in checkpoints if p.name != 'last.ckpt'} == {1, 2, 3}
    records = [json.loads(line) for line in (run / 'progress.jsonl').read_text().splitlines()]
    assert len(records) == 5  # Sanity validation must not become an epoch record.
    assert [r['epoch'] for r in records if r['new_minimum_val_loss']] == [0, 1]
    best = torch.load(next(p for p in checkpoints if p.name.startswith('epoch=1')), weights_only=True)
    assert any(state.get('best_val_loss') == 1. for state in best['callbacks'].values())
    logger = ProtocolLogger(run)
    logger.load_state_dict({'best_val_loss': 1.})
    assert logger.state_dict() == {'best_val_loss': 1.}
