#!/usr/bin/env python3
"""
Auto-labeling for medical symptoms with character spans.

- Loads data from data/symptoms.json and data/synonyms.json
- Performs longest-first, word-boundary span extraction without overlaps
- Saves labeled dataset purely to data/auto_labeled.jsonl
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Tuple
import pandas as pd

# Add the project root to path to import from chatbot module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from chatbot.symptom_extractor import load_symptom_data, canonicalize_symptom

DATA_DIR_DEFAULT = os.path.join(project_root, "data")
DEFAULT_INPUT = os.path.join(DATA_DIR_DEFAULT, "ai-medical-chatbot.csv")


def extract_character_spans(
    text: str, symptom_list: List[str], synonym_map: Dict[str, str]
) -> List[Tuple[int, int, str]]:
    """
    Return list of (start, end, canonical_label) using longest-first matching.
    """
    if not text:
        return []

    # Build phrase -> canonical label dictionary
    all_phrases: Dict[str, str] = {}
    for s in symptom_list:
        all_phrases[canonicalize_symptom(s)] = s
    for syn, canon in synonym_map.items():
        all_phrases[canonicalize_symptom(syn)] = canon

    # Sort by length desc to prioritize longer phrases
    phrases = sorted(all_phrases.items(), key=lambda x: len(x[0]), reverse=True)

    spans: List[Tuple[int, int, str]] = []
    used = set()
    lower = text.lower()

    for phrase, canon in phrases:
        if not phrase or len(phrase) < 3:
            continue
        start = 0
        while True:
            pos = lower.find(phrase, start)
            if pos == -1:
                break
            end_pos = pos + len(phrase)

            # word boundary check
            left_ok = (pos == 0) or (not text[pos - 1].isalnum())
            right_ok = (end_pos == len(text)) or (not text[end_pos].isalnum())
            if left_ok and right_ok:
                rng = set(range(pos, end_pos))
                if not rng.intersection(used):
                    spans.append((pos, end_pos, canon))
                    used.update(rng)
            start = pos + 1

    spans.sort(key=lambda x: x[0])
    return spans


def auto_label_sentences(
    csv_file: str, data_dir: str = DATA_DIR_DEFAULT, max_sentences: int = 200
) -> str:
    """
    Process CSV and write JSONL to data/auto_labeled.jsonl, returning its path.
    """
    symptoms, synonyms = load_symptom_data(data_dir)
    output_file = os.path.join(data_dir, "auto_labeled_sample.jsonl")

    chunk_size = 1000
    processed = 0
    labeled: List[dict] = []

    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        if processed >= max_sentences:
            break
        for _, row in chunk.iterrows():
            if processed >= max_sentences:
                break
            for col in ["Description", "Patient", "Doctor"]:
                if col in row and pd.notna(row[col]):
                    text = str(row[col]).strip()
                    if text and len(text) > 10:
                        spans = extract_character_spans(text, symptoms, synonyms)
                        labeled.append(
                            {
                                "text": text,
                                "spans": spans,
                                "source_column": col,
                                "row_index": int(getattr(row, "name", processed)),
                            }
                        )
                        processed += 1
                        if processed >= max_sentences:
                            break

    os.makedirs(data_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in labeled:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    total_spans = sum(len(e["spans"]) for e in labeled)
    sentences_with_spans = sum(1 for e in labeled if e["spans"])
    print(f"Total sentences: {len(labeled)} | With spans: {sentences_with_spans} | Total spans: {total_spans}")

    return output_file


def validate_spans(jsonl_file: str, num_examples: int = 5) -> None:
    """
    Print sample lines with inline highlighting and span tuples.
    """
    shown = 0
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if shown >= num_examples:
                break
            entry = json.loads(line.strip())
            text = entry["text"]
            spans = entry["spans"]
            if not spans:
                continue
            print(f"\nExample {shown + 1}")
            print(f"Text: {text}")
            print(f"Spans: {spans}")
            # show highlighted
            highlighted = text
            for start, end, label in sorted(spans, key=lambda x: x[0], reverse=True):
                highlighted = highlighted[:start] + f"[{text[start:end]}]({label})" + highlighted[end:]
            print(f"Highlighted: {highlighted}")
            shown += 1


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label medical symptoms with character spans (writes to data/auto_labeled.jsonl)"
    )
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT, help="Input CSV (default: data/ai-medical-chatbot.csv)")
    parser.add_argument("-d", "--data-dir", default=DATA_DIR_DEFAULT, help="Data dir with JSON (default: data)")
    parser.add_argument("-m", "--max-sentences", type=int, default=200, help="Max sentences (default: 200)")
    parser.add_argument("-v", "--validate", action="store_true", help="Print validation examples")
    parser.add_argument("-e", "--examples", type=int, default=3, help="Number of examples to print")

    args = parser.parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data dir not found: {args.data_dir}")
    for name in ["symptoms.json", "synonyms.json"]:
        if not os.path.exists(os.path.join(args.data_dir, name)):
            raise FileNotFoundError(f"Missing {name} in {args.data_dir}")

    print("Running auto-label...")
    out = auto_label_sentences(args.input, data_dir=args.data_dir, max_sentences=args.max_sentences)
    if args.validate:
        validate_spans(out, num_examples=args.examples)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
