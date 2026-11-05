"""Streamlit UI.  Run: streamlit run app.py   (train a model first)"""
from pathlib import Path

import streamlit as st

from generate import generate_tokens, load_all, save_midi, save_wav

st.set_page_config(page_title="AI Music Generator", page_icon="🎵")
st.title("🎵 AI Music Generator")
st.caption("An LSTM trained on MIDI music composes new pieces.")

if not Path("model/model.keras").exists():
    st.error("No trained model found. Run `python preprocess.py` then `python train.py` first.")
    st.stop()


@st.cache_resource
def load():
    return load_all()


model, vocab, tokens = load()

length = st.slider("Length (notes)", 50, 400, 150, 10)
temperature = st.slider("Creativity (temperature)", 0.3, 1.5, 0.9, 0.1,
                        help="Low = safe and repetitive, high = surprising")
speed = st.slider("Seconds per note", 0.15, 0.6, 0.3, 0.05)

if st.button("Generate music", type="primary"):
    with st.spinner("Composing..."):
        out = generate_tokens(model, vocab, tokens, length, temperature)
        Path("output").mkdir(exist_ok=True)
        midi = save_midi(out, "output/generated.mid", step=0.5)
        wav = save_wav(out, "output/generated.wav", speed)
    st.audio(Path(wav).read_bytes(), format="audio/wav")
    st.download_button("Download MIDI", Path(midi).read_bytes(), "generated.mid", "audio/midi")
    st.download_button("Download WAV", Path(wav).read_bytes(), "generated.wav", "audio/wav")
