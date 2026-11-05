"""Step 4: generate music.   python generate.py --length 200 --temperature 0.9
Creates output/generated.mid and output/generated.wav"""
import argparse
import json
from pathlib import Path

import numpy as np

from utils import midi_to_freq, render_wav, sample_with_temperature, token_to_midis


def load_all(model_path="model/model.keras", vocab_path="model/vocab.json",
             tokens_path="data/tokens.json"):
    from tensorflow import keras  # imported here so tests don't need TensorFlow
    model = keras.models.load_model(model_path)
    vocab = json.loads(Path(vocab_path).read_text())
    tokens = json.loads(Path(tokens_path).read_text())
    return model, vocab, tokens


def generate_tokens(model, vocab, tokens, length=200, temperature=1.0, seed=None):
    rng = np.random.default_rng(seed)
    index = {t: i for i, t in enumerate(vocab)}
    seq_len = model.input_shape[1]
    ids = [index[t] for t in tokens if t in index]
    start = int(rng.integers(0, len(ids) - seq_len))
    window, out = ids[start:start + seq_len], []
    for _ in range(length):
        probs = model(np.array([window], dtype="int32"), training=False).numpy()[0]
        nxt = sample_with_temperature(probs, temperature, rng)
        out.append(vocab[nxt])
        window = window[1:] + [nxt]
    return out


def save_midi(tokens, path, step=0.5):
    from music21 import chord, note, stream
    s = stream.Stream()
    for i, tok in enumerate(tokens):
        midis = token_to_midis(tok)
        el = note.Note(midis[0]) if len(midis) == 1 else chord.Chord(midis)
        el.quarterLength = step
        s.insert(i * step, el)
    s.write("midi", fp=str(path))
    return str(path)


def save_wav(tokens, path, seconds_per_note=0.3):
    events = [([midi_to_freq(m) for m in token_to_midis(t)], i * seconds_per_note, seconds_per_note * 1.6)
              for i, t in enumerate(tokens)]
    return render_wav(events, path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="output/generated")
    a = ap.parse_args()

    model, vocab, tokens = load_all()
    out = generate_tokens(model, vocab, tokens, a.length, a.temperature, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    print("MIDI:", save_midi(out, a.out + ".mid"))
    print("WAV :", save_wav(out, a.out + ".wav"))
