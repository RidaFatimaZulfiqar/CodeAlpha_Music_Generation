import wave
import numpy as np
import pytest
import utils as u


def test_build_sequences_shapes():
    toks = ["60", "62", "64", "60", "62", "64", "65", "67"]
    X, y, vocab = u.build_sequences(toks, seq_len=3)
    assert X.shape == (5, 3) and y.shape == (5,)
    assert vocab == sorted(set(toks))
    assert vocab[y[0]] == toks[3]


def test_not_enough_data():
    with pytest.raises(ValueError):
        u.build_sequences(["60", "62"], seq_len=5)


def test_sampling():
    p = [0.1, 0.7, 0.2]
    assert u.sample_with_temperature(p, 0) == 1
    rng = np.random.default_rng(0)
    picks = [u.sample_with_temperature(p, 0.2, rng) for _ in range(200)]
    assert picks.count(1) > 150          # low temperature stays on the likeliest note
    picks = [u.sample_with_temperature(p, 3.0, rng) for _ in range(300)]
    assert len(set(picks)) == 3          # high temperature explores


def test_tokens_and_freq():
    assert u.token_to_midis("60") == [60]
    assert u.token_to_midis("60.64.67") == [60, 64, 67]
    assert abs(u.midi_to_freq(69) - 440) < 1e-9


def test_render_wav(tmp_path):
    p = u.render_wav([([261.6], 0, 0.3), ([261.6, 329.6, 392.0], 0.3, 0.3)], tmp_path / "t.wav")
    with wave.open(p) as w:
        assert w.getnchannels() == 1 and w.getnframes() > 10000
