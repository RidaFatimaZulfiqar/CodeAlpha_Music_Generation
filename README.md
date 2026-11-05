# CodeAlpha_Music_Generation 🎵

An AI music generator built for the **CodeAlpha AI Internship (Task 3)**. An **LSTM** neural network learns note patterns from MIDI music and then composes new pieces, which you can play or save as **MIDI** and **WAV**.

**Live demo (runs in your browser):** https://ridafatimazulfiqar.github.io/CodeAlpha_Music_Generation/

## Features
- Collects MIDI data: Bach chorales built into `music21`, or your own `.mid` files
- Preprocessing with **music21**: notes and chords become tokens (MIDI pitch numbers)
- Deep learning model: **Embedding + 2-layer LSTM** in Keras/TensorFlow
- **Temperature sampling** to control how safe or creative the music is
- Saves generated music as **MIDI** (`.mid`) and **audio** (`.wav`, synthesised with NumPy, no extra software)
- **Streamlit app** to generate and play music
- **Browser version** (`index.html`): trains a small LSTM with TensorFlow.js, then plays, shows a piano roll and exports MIDI

## How it works
```
MIDI files -> music21 -> note/chord tokens -> sequences of 32 tokens
   -> LSTM learns "what note comes next" -> start from a seed, predict one note at a time
   -> sample with temperature -> MIDI + WAV
```

## Project structure
```
CodeAlpha_Music_Generation/
├── preprocess.py   # Step 1: MIDI -> tokens (music21)
├── model.py        # Step 2: LSTM model
├── train.py        # Step 3: training
├── generate.py     # Step 4: generate MIDI + WAV
├── app.py          # Streamlit app
├── utils.py        # Sequences, sampling, WAV synthesis
├── test_music.py   # Tests
├── index.html      # Browser version (GitHub Pages)
├── data/ model/ output/
└── requirements.txt
```

## Setup
```bash
git clone https://github.com/<your-username>/CodeAlpha_Music_Generation.git
cd CodeAlpha_Music_Generation
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Use Python 3.9 to 3.12 (TensorFlow does not support the newest Python versions straight away).

## Run (in this order)
```bash
python preprocess.py --source corpus --limit 40   # 1. collect + preprocess (Bach chorales)
python train.py --epochs 30                       # 2. train (a few minutes on CPU)
python generate.py --length 200 --temperature 0.9 # 3. creates output/generated.mid and .wav
streamlit run app.py                              # or use the web app
```
To use your own music, put `.mid` files in `data/midi/` and run `python preprocess.py --source folder --folder data/midi`.

**Tips:** more files and more epochs sound better. Temperature around 0.7 is steadier, and around 1.2 is more experimental. Open the `.mid` file in any MIDI player, MuseScore or GarageBand.

## Browser version
`index.html` trains a small LSTM on six classic melodies (transposed into 12 keys) in your browser, then generates and plays new melodies with the Web Audio API. It also draws a piano roll and lets you download the result as a MIDI file. Open it from GitHub Pages or from `localhost`.

## Limitations and ideas
All notes have equal length, so rhythm is simple. Ideas: add note durations, train on a single composer or genre, use a GAN or Transformer, add multiple instruments.

## Tech stack
Python, TensorFlow/Keras, music21, NumPy, Streamlit, TensorFlow.js

---
Made for the CodeAlpha AI Internship.
