"""Step 3: train the model.   python train.py --epochs 30"""
import argparse
import json
from pathlib import Path

from tensorflow import keras

from model import build_model
from utils import build_sequences

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", default="data/tokens.json")
ap.add_argument("--seq-len", type=int, default=32)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=64)
args = ap.parse_args()

tokens = json.loads(Path(args.tokens).read_text())
X, y, vocab = build_sequences(tokens, args.seq_len)
print(f"{len(X)} training sequences, vocabulary size {len(vocab)}")

Path("model").mkdir(exist_ok=True)
Path("model/vocab.json").write_text(json.dumps(vocab))

model = build_model(args.seq_len, len(vocab))
model.fit(
    X, y, epochs=args.epochs, batch_size=args.batch, validation_split=0.1,
    callbacks=[
        keras.callbacks.ModelCheckpoint("model/model.keras", save_best_only=True, monitor="val_loss"),
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    ],
)
model.save("model/model.keras")
print("Saved model/model.keras")
