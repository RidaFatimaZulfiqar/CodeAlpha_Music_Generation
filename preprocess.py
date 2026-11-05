"""Step 1: collect MIDI data and turn it into note tokens.

Examples:
  python preprocess.py --source corpus --limit 40      # Bach chorales built into music21
  python preprocess.py --source folder --folder data/midi   # your own .mid files
Each note or chord becomes a token such as '60' or '60.64.67' (MIDI pitch numbers).
"""
import argparse
import glob
import json
from pathlib import Path

from music21 import converter, corpus


def stream_to_tokens(score):
    tokens = []
    for c in score.chordify().recurse().getElementsByClass("Chord"):
        tokens.append(".".join(str(p.midi) for p in c.pitches))
    return tokens


def collect_paths(args):
    if args.source == "corpus":
        paths = [p for p in corpus.getComposer(args.composer)
                 if str(p).lower().endswith((".mxl", ".xml", ".mid", ".midi", ".krn"))]
    else:
        paths = glob.glob(str(Path(args.folder) / "**" / "*.mid*"), recursive=True)
    return paths[: args.limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["corpus", "folder"], default="corpus")
    ap.add_argument("--composer", default="bach", help="composer for --source corpus")
    ap.add_argument("--folder", default="data/midi")
    ap.add_argument("--limit", type=int, default=40, help="max number of files")
    ap.add_argument("--out", default="data/tokens.json")
    args = ap.parse_args()

    paths = collect_paths(args)
    if not paths:
        raise SystemExit("No MIDI/music files found. Check --folder or --source.")
    print(f"Parsing {len(paths)} files...")

    tokens = []
    for i, p in enumerate(paths, 1):
        try:
            tokens += stream_to_tokens(converter.parse(str(p)))
            print(f"  [{i}/{len(paths)}] {Path(str(p)).name}  (total tokens: {len(tokens)})")
        except Exception as e:  # skip unreadable files
            print(f"  skipped {p}: {e}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(tokens))
    print(f"Saved {len(tokens)} tokens ({len(set(tokens))} unique) to {args.out}")


if __name__ == "__main__":
    main()
