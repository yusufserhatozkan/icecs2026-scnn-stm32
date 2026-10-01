"""Reject inconsistent paper tables, archived metrics and checkpoint selections."""
import csv
import json
import shutil

import pytest

from tools import verify_artifacts as verify


@pytest.fixture
def archive(tmp_path, monkeypatch):
    source = verify.ROOT
    shutil.copytree(source / 'results', tmp_path / 'results')
    (tmp_path / 'models').mkdir()
    shutil.copyfile(source / 'models/manifest.json', tmp_path / 'models/manifest.json')
    monkeypatch.setattr(verify, 'ROOT', tmp_path)
    return tmp_path


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding='utf-8')


def write_csv(path, records):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def test_archived_tables_quality_and_training_are_consistent(archive):
    verify.verify_quality(verify.verify_tables())
    verify.verify_training()


def test_reference_aggregate_must_round_to_paper(archive):
    path = archive / 'results/metrics/n256_reference.json'
    value = json.loads(path.read_text())
    value['metrics']['enh_sisnr'] += 1
    write_json(path, value)
    with pytest.raises(ValueError, match='does not round to Table I'):
        verify.verify_quality(verify.verify_tables())


def test_baseline_mean_must_match_individual_records(archive):
    path = archive / 'results/per_utterance/n64.csv'
    records = verify.read_csv('results/per_utterance/n64.csv')
    records[0]['enh_sisnr'] = float(records[0]['enh_sisnr']) + 1
    write_csv(path, records)
    with pytest.raises(ValueError, match='mean does not match'):
        verify.verify_quality(verify.verify_tables())


def test_noisy_alignment_checked_even_when_means_match(archive):
    path = archive / 'results/per_utterance/n64.csv'
    records = verify.read_csv('results/per_utterance/n64.csv')
    for key in ('noisy_sisnr', 'noisy_pesq', 'noisy_stoi'):
        records[0][key], records[1][key] = records[1][key], records[0][key]
    write_csv(path, records)
    with pytest.raises(ValueError, match='noisy scores differ'):
        verify.verify_quality(verify.verify_tables())


def test_selection_must_be_actual_validation_minimum(archive):
    path = archive / 'results/training/summary.json'
    value = json.loads(path.read_text())
    value['models'][0]['selected_epoch_zero_based'] += 1
    write_json(path, value)
    with pytest.raises(ValueError, match='validation minimum'):
        verify.verify_training()


def test_training_metadata_rejects_speaker_overlap(archive):
    path = archive / 'results/training/summary.json'
    value = json.loads(path.read_text())
    value['splits']['train']['speakers'].append('p232')
    write_json(path, value)
    with pytest.raises(ValueError, match='Speaker overlap'):
        verify.verify_training()


def test_paper_table_is_not_silently_replaced_with_graph_count(archive):
    path = archive / 'results/table2.csv'
    records = verify.read_csv('results/table2.csv')
    records[2]['n128'] = '44'
    write_csv(path, records)
    with pytest.raises(ValueError, match='Table II transcription'):
        verify.verify_tables()
