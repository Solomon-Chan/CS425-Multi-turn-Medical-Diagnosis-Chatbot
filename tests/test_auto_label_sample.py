#!/usr/bin/env python3
"""
Test runner for auto_label.py - runs over a small sample (100–200 sentences) to validate spans.
Located in tests/; resolves imports relative to project root and scripts/ directory.
"""

import os
import json
import sys
import pandas as pd

# Resolve project root (parent of tests/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import symptom data loader from chatbot module
from chatbot.symptom_extractor import load_symptom_data

# Ensure scripts/ is importable for auto_label
scripts_dir = os.path.join(project_root, "scripts")
sys.path.insert(0, scripts_dir)
import auto_label

DATA_DIR_DEFAULT = os.path.join(project_root, "data")
DEFAULT_INPUT = os.path.join(DATA_DIR_DEFAULT, "ai-medical-chatbot.csv")


def ensure_data_dir(path=None):
    if path is None:
        path = DATA_DIR_DEFAULT
    os.makedirs(path, exist_ok=True)
    return path


def build_or_find_sample_csv(data_dir=None, n=150):
    if data_dir is None:
        data_dir = DATA_DIR_DEFAULT
    csv_path = os.path.join(data_dir, "ai-medical-chatbot.csv")
    if os.path.exists(csv_path):
        return csv_path

    # Fallback: synthesize a small CSV of n rows using known symptoms/synonyms
    symptoms, synonyms = load_symptom_data(data_dir)
    sample_texts = []
    # Mix canonical and synonym phrases
    canon_samples = symptoms[: min(30, len(symptoms))]
    syn_samples = list(synonyms.keys())[: min(30, len(synonyms))]
    seed = []
    for c in canon_samples:
        seed.append(f"I am experiencing {c} recently and need advice.")
    for s in syn_samples:
        seed.append(f"My doctor noted {s} and suggested rest.")
    # Repeat until n rows
    while len(sample_texts) < n:
        sample_texts.extend(seed)
    sample_texts = sample_texts[:n]

    df = pd.DataFrame(
        {
            "Description": sample_texts,
            "Patient": [None] * n,
            "Doctor": [None] * n,
        }
    )
    out_csv = os.path.join(data_dir, "sample_auto_label.csv")
    df.to_csv(out_csv, index=False)
    print(f"Created synthetic CSV with {n} rows: {out_csv}")
    return out_csv


def test_small_sample_end_to_end():
    print("Running auto-label test on small sample...")
    data_dir = ensure_data_dir(DATA_DIR_DEFAULT)
    csv_file = build_or_find_sample_csv(data_dir, n=150)
    out_path = auto_label.auto_label_sentences(csv_file, data_dir=data_dir, max_sentences=150)

    # Basic validations on first few lines
    symptoms, _ = load_symptom_data(data_dir)
    validation_count = 0
    with open(out_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            entry = json.loads(line)
            text = entry["text"]
            spans = entry["spans"]
            # Span tuple integrity
            for s, e, label in spans:
                assert 0 <= s < e <= len(text), f"Invalid span bounds: {(s, e)} in text of length {len(text)}"
                assert label in symptoms, f"Label not canonical: {label}"
                validation_count += 1

    print(f"✓ Validated {validation_count} spans from first 10 entries")

    # Print a few examples to eyeball spans
    print("\n=== Sample outputs for manual validation ===")
    auto_label.validate_spans(out_path, num_examples=3)
    print(f"\n✓ Test completed successfully! Output saved to: {out_path}")


if __name__ == "__main__":
    test_small_sample_end_to_end()
