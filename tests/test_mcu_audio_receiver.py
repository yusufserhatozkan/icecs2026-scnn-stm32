"""Host capture checks using synthetic audio and serial byte streams."""
import io
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools import mcu_audio_receiver as receiver


def test_sample_comparison_detects_gain_change_hidden_by_sisnr():
    rng = np.random.default_rng(42)
    clean = rng.standard_normal(80).astype(np.float32)
    reference = clean + 0.1 * rng.standard_normal(80).astype(np.float32)
    audio = 2 * reference
    assert abs(receiver.si_snr(audio, clean) - receiver.si_snr(reference, clean)) < 1e-4
    result = receiver.compare_waveforms(audio, reference)
    assert result['sample_count'] == 80
    assert result['rmse'] > 0.5
    assert result['max_abs_error'] > 1
    assert receiver.compare_waveforms(reference, reference)['rmse'] == 0


@pytest.mark.parametrize('audio,reference', [([], []), ([1], [1, 2]),
                                           ([[1]], [1]), ([np.nan], [0]),
                                           ([0], [np.inf])])
def test_invalid_comparisons_fail(audio, reference):
    with pytest.raises(ValueError, match='Waveforms must'):
        receiver.compare_waveforms(np.asarray(audio), np.asarray(reference))


def test_frame_count_comes_from_binary_and_checks_sidecar(tmp_path):
    binary = tmp_path / 'reference.bin'
    binary.write_bytes(np.zeros(80, dtype='<f4').tobytes())
    assert receiver.load_n_frames(str(binary)) == 2
    (tmp_path / 'reference_sisnr.txt').write_text('n_frames: 3\n')
    with pytest.raises(ValueError, match='frame count'):
        receiver.load_n_frames(str(binary))


def test_truncated_serial_capture_fails_instead_of_waiting_forever():
    with pytest.raises(TimeoutError, match='2 of 4 bytes'):
        receiver.read_audio_bytes(io.BytesIO(b'12'), 4)
    assert receiver.read_audio_bytes(io.BytesIO(b'1234'), 4) == b'1234'


def test_raw_samples_are_preserved_and_capture_cannot_be_overwritten(tmp_path):
    rng = np.random.default_rng(19)
    clean = rng.standard_normal(80).astype(np.float32)
    reference = clean + 0.1 * rng.standard_normal(80).astype(np.float32)
    audio = reference * 2
    target = tmp_path / 'capture.wav'
    result = receiver.save_capture(target, audio, reference, clean, 'synthetic')
    np.testing.assert_array_equal(np.fromfile(target.with_suffix('.bin'), dtype='<f4'), audio)
    report = (tmp_path / 'mcu_test_results.txt').read_text()
    assert f"rmse: {result['rmse']}" in report
    assert 'sample_count: 80' in report
    before = {p: p.read_bytes() for p in receiver.capture_paths(target)}
    with pytest.raises(FileExistsError):
        receiver.save_capture(target, audio * 3, reference, clean, 'synthetic')
    assert all(p.read_bytes() == data for p, data in before.items())


def test_receiver_cli_reads_framed_serial_bytes_and_writes_comparison(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    clean = rng.standard_normal(80).astype(np.float32)
    reference = clean + 0.1 * rng.standard_normal(80).astype(np.float32)
    audio = reference + 0.002
    clean_path, ref_path = tmp_path / 'clean.wav', tmp_path / 'reference.bin'
    receiver.sf.write(clean_path, clean, 16000, subtype='FLOAT')
    ref_path.write_bytes(reference.astype('<f4').tobytes())
    output = tmp_path / 'capture' / 'enhanced.wav'

    class SerialCapture(io.BytesIO):
        def __init__(self):
            super().__init__(audio.astype('<f4').tobytes())
            self.lines = iter([b'AUDIO_START\r\n', b'TIMING:2.715 ms/frame\r\n',
                               b'AUDIO_END\r\n'])

        def readline(self):
            return next(self.lines, b'')

    connection = SerialCapture()
    monkeypatch.setitem(sys.modules, 'serial', SimpleNamespace(Serial=lambda *a, **k: connection))
    monkeypatch.setattr(receiver.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(sys, 'argv', ['receiver', '--clean', str(clean_path),
                                     '--ref_bin', str(ref_path), '--out', str(output)])
    receiver.main()
    assert connection.closed
    np.testing.assert_array_equal(np.fromfile(output.with_suffix('.bin'), dtype='<f4'), audio)
    report = (output.parent / 'mcu_test_results.txt').read_text()
    assert 'sample_count: 80' in report
    assert 'TIMING:2.715 ms/frame' in report
