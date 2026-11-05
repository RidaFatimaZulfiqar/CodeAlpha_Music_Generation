"""Step 2: the LSTM network."""
from tensorflow import keras
from tensorflow.keras import layers


def build_model(seq_len, vocab_size):
    model = keras.Sequential([
        layers.Input(shape=(seq_len,), dtype="int32"),
        layers.Embedding(vocab_size, 64),
        layers.LSTM(256, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(256),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(vocab_size, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    return model
