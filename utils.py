"""Pure NumPy helpers (no TensorFlow / music21 needed). Easy to test."""
import wave

import numpy as np


def build_sequences(tokens, seq_len=32):
    """Turn a token list into (X, y, vocab) for next-token prediction."""
    vocab = sorted(set(tokens))
    index = {t: i for i, t in enumerate(vocab)}
    ids = np.array([index[t] for t in tokens], dtype="int32")
    n = len(ids) - seq_len
    if n <= 0:
        raise ValueError("Not enough notes: need more than seq_len tokens.")
    X = np.stack([ids[i:i + seq_len] for i in range(n)])
    y = ids[seq_len:]
    return X, y, vocab


def sample_with_temperature(probs, temperature=1.0, rng=None):
    """Pick the next token. Low temperature = safe/repetitive, high = adventurous."""
    rng = rng or np.random.default_rng()
    probs = np.asarray(probs, dtype=float)
    if temperature <= 0:
        return int(np.argmax(probs))
    logits = np.log(probs + 1e-9) / temperature
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def token_to_midis(token):
    """'60' -> [60];  '60.64.67' -> [60, 64, 67] (a chord)."""
    return [int(x) for x in str(token).split(".")]


def midi_to_freq(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def render_wav(events, path, sample_rate=22050):
    """events: list of (list_of_frequencies, start_seconds, duration_seconds).
    Synthesises a soft piano-like tone and saves a 16-bit mono WAV."""
    if not events:
        raise ValueError("No notes to render.")
    total = max(s + d for _, s, d in events) + 0.6
    audio = np.zeros(int(total * sample_rate))
    for freqs, start, dur in events:
        n = int(dur * sample_rate)
        i = int(start * sample_rate)
        n = min(n, len(audio) - i)
        if n <= 0:
            continue
        t = np.arange(n) / sample_rate
        env = np.minimum(1.0, t / 0.01) * np.exp(-2.5 * t / max(dur, 1e-3))
        tone = sum(np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) for f in freqs)
        audio[i:i + n] += tone / max(len(freqs), 1) * env
    audio /= max(1.0, np.abs(audio).max())
    pcm = (audio * 0.8 * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return str(path)
