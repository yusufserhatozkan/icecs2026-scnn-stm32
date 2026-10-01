"""Check the packaged files and reported tables without running experiments."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def read_csv(name):
    with (ROOT / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_tables():
    table = read_csv('results/table1.csv')
    expected_rows = [
        ('Noisy input', '8.44', '1.971', '0.921'),
        ('Reproduced SCNN N=256', '18.08', '2.264', '0.925'),
        ('SCNN N=128, ONNX streaming', '17.42', '2.149', '0.923'),
        ('SCNN N=64, ONNX streaming', '16.70', '2.022', '0.922'),
    ]
    if [tuple(row.values()) for row in table] != expected_rows:
        raise ValueError('Table I differs from the paper')
    expected_table2 = [
        ('Weights (Flash), KB', '280', '91'),
        ('Activations (SRAM), KB', '46', '23'),
        ('ONNX graph nodes', '43', '43'),
        ('Analyze wall-time, s', '16', '16'),
    ]
    if [tuple(row.values()) for row in read_csv('results/table2.csv')] != expected_table2:
        raise ValueError('Table II transcription differs from the paper')
    deployment = read_csv('results/deployment.csv')
    expected_deployment = [
        ('N=128', '160', '1662', '2.5', '6.145', '2.46', '15.66'),
        ('N=64', '160', '1662', '2.5', '2.715', '1.086', '15.98'),
    ]
    if [tuple(row.values()) for row in deployment] != expected_deployment:
        raise ValueError('Deployment transcription differs from the paper')
    for row, precision in zip(deployment, (2, 3)):
        calculated = float(row['latency_ms_per_frame']) / (40 / 16000 * 1000)
        if f'{calculated:.{precision}f}' != row['rtf_reported']:
            raise ValueError('Reported RTF does not follow from latency and hop')
    return table


def verify_quality(table):
    ids = []
    noisy_records = []
    required = {f'{side}_{metric}' for side in ('noisy', 'enh')
                for metric in ('sisnr', 'pesq', 'stoi')}
    for name, table_row in (('n256_reference', 1), ('n128', 2), ('n64', 3)):
        summary = read_json(f'results/metrics/{name}.json')
        if summary['n_utterances'] != 824 or set(summary['metrics']) != required:
            raise ValueError(f'{name}: invalid summary coverage')
        for side, row_index in (('noisy', 0), ('enh', table_row)):
            for key, column, digits in (('sisnr', 'si_snr_db', 2),
                                        ('pesq', 'pesq_wb', 3), ('stoi', 'stoi', 3)):
                value = summary['metrics'][f'{side}_{key}']
                if not math.isfinite(value) or f'{value:.{digits}f}' != table[row_index][column]:
                    raise ValueError(f'{name}/{side}_{key}: does not round to Table I')
        if name == 'n256_reference':
            if summary['per_utterance_records_available'] is not False:
                raise ValueError('N=256 per-utterance coverage is not established')
            continue
        records = read_csv(f'results/per_utterance/{name}.csv')
        if len(records) != 824 or len({r['id'] for r in records}) != 824:
            raise ValueError(f'{name}: invalid utterance coverage')
        if any(int(r['length_samples']) <= 0 for r in records):
            raise ValueError(f'{name}: invalid utterance length')
        ids.append([(r['id'], r['length_samples']) for r in records])
        noisy_records.append([tuple(r[f'noisy_{m}'] for m in ('sisnr', 'pesq', 'stoi'))
                              for r in records])
        for metric, expected in summary['metrics'].items():
            values = [float(r[metric]) for r in records]
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f'{name}/{metric}: nonfinite value')
            if abs(statistics.fmean(values) - expected) > 1e-12:
                raise ValueError(f'{name}/{metric}: mean does not match saved records')
            if summary['successful_counts'][metric] != 824:
                raise ValueError(f'{name}/{metric}: incomplete metric count')
    if ids[0] != ids[1]:
        raise ValueError('Baseline utterance IDs or lengths differ')
    if noisy_records[0] != noisy_records[1]:
        raise ValueError('Baseline noisy scores differ')


def verify_training():
    summary = read_json('results/training/summary.json')
    models = {m['model']: m for m in read_json('models/manifest.json')['models']}
    if {m['model'] for m in summary['models']} != {'n128', 'n64'}:
        raise ValueError('Unexpected training model set')
    for item in summary['models']:
        path = ROOT / item['progress']
        if sha256(path) != item['progress_sha256']:
            raise ValueError('Training progress hash mismatch')
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if item['epochs_completed'] != 100 or [r['epoch'] for r in records] != list(range(100)):
            raise ValueError('Incomplete training epoch coverage')
        if not all(math.isfinite(r['val_loss']) for r in records):
            raise ValueError('Nonfinite validation loss')
        selected = min(records, key=lambda r: r['val_loss'])
        model = models[item['model']]
        if (selected['epoch'] != item['selected_epoch_zero_based']
                or selected['val_loss'] != item['selected_validation_loss']
                or selected['epoch'] != model['source_epoch_zero_based']
                or item['source_checkpoint_sha256'] != model['source_checkpoint_sha256']):
            raise ValueError('Selected checkpoint is not the recorded validation minimum')
    splits = summary['splits']
    if {name: row['pairs'] for name, row in splits.items()} != {
            'train': 10802, 'valid': 770, 'test': 824}:
        raise ValueError('Split counts differ from the paper')
    speakers = [set(splits[name]['speakers']) for name in ('train', 'valid', 'test')]
    if any(speakers[i] & speakers[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Speaker overlap in split metadata')


def verify_board_archive():
    for name, samples in (('test_clean', 66522), ('test_noisy', 66522),
                          ('test_enhanced_ref', 66480), ('mcu_enhanced', 66480)):
        with wave.open(str(ROOT / f'deploy/n64/{name}.wav'), 'rb') as stream:
            actual = (stream.getframerate(), stream.getnchannels(),
                      stream.getsampwidth(), stream.getnframes())
        if actual != (16000, 1, 2, samples):
            raise ValueError(f'{name}: unexpected saved WAV format or length')
    if (ROOT / 'deploy/n64/test_enhanced_ref.bin').stat().st_size != 1662 * 40 * 4:
        raise ValueError('Reference binary length differs from the recorded frame count')
    for name, weights, activations in (('n128', 285216, 47072), ('n64', 93472, 23264)):
        report = (ROOT / f'results/compiler/{name}.txt').read_text(encoding='utf-8')
        for label, value in (('weights size', weights), ('activations size', activations),
                             ('c-node #', 43)):
            match = re.search(re.escape(label) + r'\s*:\s*(\d+)', report)
            if not match or int(match.group(1)) != value:
                raise ValueError(f'{name}: compiler {label} changed')
        digest = re.search(r'model_hash\s*:\s*0x([a-f0-9]+)', report).group(1)
        graph = ROOT / f'models/dpsnn_{name}_streaming_xcubeai.onnx'
        if hashlib.md5(graph.read_bytes()).hexdigest() != digest:
            raise ValueError(f'{name}: graph differs from compiler input')


def verify_models():
    import numpy as np
    import onnx
    from onnx import numpy_helper
    import torch
    sys.path.insert(0, str(ROOT))
    from export.export_to_onnx import load_from_checkpoint
    manifest = read_json('models/manifest.json')
    for item in manifest['models']:
        path = ROOT / item['path']
        if sha256(path) != item['sha256']:
            raise ValueError('Checkpoint hash differs from model manifest')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if set(checkpoint) != {'state_dict', 'hyper_parameters'}:
            raise ValueError('Unexpected checkpoint payload')
        if checkpoint['hyper_parameters'] != item['hyper_parameters']:
            raise ValueError('Checkpoint hyperparameters differ from model manifest')
        if len(checkpoint['state_dict']) != item['state_tensors']:
            raise ValueError('Checkpoint tensor count mismatch')
        if not all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values()):
            raise ValueError('Nonfinite checkpoint tensor')
        if item['model'] == 'n256_reference':
            if 'readout_threshold' not in checkpoint['state_dict']:
                raise ValueError('N=256 legacy archive changed')
            continue
        model = load_from_checkpoint(str(path))
        if (not model.scnn_only or model.no_pointwise or model.L != 80
                or model.stride != 40 or model.context_step != 4):
            raise ValueError('Baseline architecture differs from paper')
        expected_count = {128: 71299, 64: 23363}[model.N]
        if sum(p.numel() for p in model.parameters()) != expected_count:
            raise ValueError('Baseline parameter count changed')
        raw = onnx.load(ROOT / f'models/dpsnn_n{model.N}_streaming.onnx')
        tensors = {i.name: numpy_helper.to_array(i) for i in raw.graph.initializer}
        for key, value in model.state_dict().items():
            onnx_name = key.replace('repeats.0.0.', 'sconv1d.')
            if onnx_name not in tensors or not np.array_equal(value.numpy(), tensors[onnx_name]):
                raise ValueError(f'{item["model"]}: ONNX initializer differs: {key}')
    for item in manifest['graphs']:
        path = ROOT / item['path']
        graph = onnx.load(path)
        onnx.checker.check_model(graph)
        if sha256(path) != item['sha256'] or len(graph.graph.node) != item['onnx_nodes']:
            raise ValueError('ONNX graph differs from manifest')
        width = 128 if '_n128_' in path.name else 64
        shapes = [[1, 80], [1, width, 4], [1, width, 1], [1, width], [1, 40]]
        for ports, expected in ((graph.graph.input, shapes),
                                (graph.graph.output, [[1, 40]] + shapes[1:])):
            actual = [[d.dim_value for d in v.type.tensor_type.shape.dim] for v in ports]
            if actual != expected or any(v.type.tensor_type.elem_type != 1 for v in ports):
                raise ValueError('Streaming port shapes or types differ from paper')
        if not any(n.op_type == 'ConvTranspose' for n in graph.graph.node):
            raise ValueError('Missing convolution decoder')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', action='store_true',
                        help='Also inspect checkpoint tensors and ONNX structure; no inference')
    args = parser.parse_args()
    manifest = read_json('artifacts.json')
    for name, expected in manifest['sha256'].items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Missing or changed artifact: {name}')
    verify_quality(verify_tables())
    verify_training()
    verify_board_archive()
    if args.models:
        verify_models()
    print(f"Verified {len(manifest['sha256'])} file hashes, paper transcriptions, "
          "both 824-utterance records, baseline training records and saved board artifacts.")
    print('Known paper discrepancies and missing evidence remain; see docs/RESULTS.md.')


if __name__ == '__main__':
    main()
