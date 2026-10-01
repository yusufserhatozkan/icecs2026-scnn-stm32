"""Check the packaged files and reported tables without running experiments."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'artifacts.json').read_text(encoding='utf-8'))
    for name, expected in manifest['sha256'].items():
        path = ROOT / name
        if not path.is_file():
            raise ValueError(f'Missing artifact: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Changed artifact: {name}')
    with (ROOT / 'results/table1.csv').open(encoding='utf-8', newline='') as stream:
        table = list(csv.DictReader(stream))
    expected_rows = [
        ('Noisy input', '8.44', '1.971', '0.921'),
        ('Reproduced SCNN N=256', '18.08', '2.264', '0.925'),
        ('SCNN N=128, ONNX streaming', '17.42', '2.149', '0.923'),
        ('SCNN N=64, ONNX streaming', '16.70', '2.022', '0.922'),
    ]
    if [tuple(row.values()) for row in table] != expected_rows:
        raise ValueError('Table I differs from the paper')
    ids = []
    for name in ('n128', 'n64'):
        summary = json.loads((ROOT / f'results/metrics/{name}.json').read_text())
        with (ROOT / f'results/per_utterance/{name}.csv').open(newline='') as stream:
            records = list(csv.DictReader(stream))
        if len(records) != 824 or len({r['id'] for r in records}) != 824:
            raise ValueError(f'{name}: invalid utterance coverage')
        ids.append([(r['id'], r['length_samples']) for r in records])
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
    print(f"Verified {len(manifest['sha256'])} files and both 824-utterance baseline records.")


if __name__ == '__main__':
    main()
