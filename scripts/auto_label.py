#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto-labeling for medical symptoms with character spans (precision-first).

Key changes:
- Restrict labeling to Description and Patient fields (skip Doctor).
- Longest-first exact matching with word boundaries, no overlaps.
- Fuzzy fallback only for anchored multiword n-grams (≥2 tokens, ≥6 chars) with
  headword gating and dynamic cutoffs; 1-grams effectively disabled.
- Lightweight negation and uncertainty scope filters (NegEx-style window).
- Synonym sanitization: drop broad, single-token, highly polysemous keys.
- Spans written as 4-tuples: (start, end, canonical_label, confidence).
"""

import argparse
import json
import os
import re
import sys
from typing import List, Dict, Tuple, Optional

import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Prefer project helpers if available; otherwise provide fallbacks
try:
    from chatbot.symptom_extractor import load_symptom_data, canonicalize_symptom
except Exception:
    def canonicalize_symptom(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    def load_symptom_data(data_dir: str):
        with open(os.path.join(data_dir, "symptoms.json"), "r", encoding="utf-8") as f:
            symptoms = json.load(f)
        with open(os.path.join(data_dir, "synonyms.json"), "r", encoding="utf-8") as f:
            synonyms = json.load(f)
        return symptoms, synonyms

# Fuzzy matching (RapidFuzz)
try:
    from rapidfuzz import process, fuzz
except ImportError as e:
    raise ImportError("rapidfuzz is required for fuzzy matching; install with `pip install rapidfuzz`") from e

DATA_DIR_DEFAULT = os.path.join(project_root, "data")
DEFAULT_INPUT = os.path.join(DATA_DIR_DEFAULT, "ai-medical-chatbot.csv")

# Headwords to gate fuzzy candidates
HEADWORDS = {
    "pain","pains","ache","aches",
    "fever","cough","rash","itch","itching",
    "urine","urination","stool","blood","vomit","nausea","diarrhea","diarrhoea",
    "breath","breathing","wheeze","fatigue","tired","dizzy","headache",
    "vision","eye","throat","nose","congestion","sneeze","cramp","swelling",
    "chest","back","abdomen","abdominal","knee","hip","neck","joint"
}

# Negation and uncertainty cues
NEG_TRIGGERS = {
    "no","not","denies","without","never","free of","absence of","negative for"
}
UNCERTAIN = {
    "possible","possibly","suspect","suggestive","might","could","likely","rule out","r/o","concern for"
}

# Broad, high-risk single-token synonym keys to prune at load
BANNED_SINGLE_TOKEN = {
    "pain","ache","hurts","sore","hot","cold","sick","weak","sad","down","tired","drained","worried","nervous"
}

# Whitelist single-token medical terms that are safe synonyms
SAFE_SINGLE_TOKEN = {
    "migraine","wheezing","coryza","jaundice","pimples","obese","vertigo","hemoptysis","lacrimation"
}

def sanitize_synonyms(syn_map: Dict[str, str]) -> Dict[str, str]:
    """Prune overly broad synonyms and keep/augment anchored phrases."""
    cleaned: Dict[str, str] = {}
    for k, v in syn_map.items():
        key = k.strip().lower()
        toks = key.split()
        if len(toks) == 1:
            # Keep explicit safe medical terms; drop polysemous singles
            if key in SAFE_SINGLE_TOKEN:
                cleaned[key] = v
            elif key in BANNED_SINGLE_TOKEN:
                continue
            else:
                # Keep domain-specific singles that look medical (heuristic)
                if len(key) >= 6 and re.search(r"[a-z]{6,}", key):
                    cleaned[key] = v
                else:
                    continue
        else:
            # Prefer anchored multiword phrases; keep
            cleaned[key] = v

    # Ensure key anchored variants exist (idempotent)
    anchored = {
        "shortness of breath": "breathlessness",
        "painful urination": "burning micturition",
        "blood in stool": "bloody stool",
        "blurry vision": "blurred and distorted vision",
        "sore throat": "throat irritation",
        "chest tightness": "chest pain",
        "sharp chest pain": "chest pain",
        "lower abdominal pain": "abdominal pain",
        "sharp abdominal pain": "abdominal pain",
    }
    for k, v in anchored.items():
        cleaned.setdefault(k, v)
    return cleaned

def build_phrase_map(symptom_list: List[str], synonym_map: Dict[str, str]) -> Dict[str, str]:
    """Build canonicalized phrase -> canonical label map (includes symptoms and synonyms)."""
    all_phrases: Dict[str, str] = {}
    for s in symptom_list:
        all_phrases[canonicalize_symptom(s)] = s
    for syn, canon in synonym_map.items():
        all_phrases[canonicalize_symptom(syn)] = canon
    return all_phrases

def contains_headword(text_slice: str) -> bool:
    toks = set(text_slice.lower().split())
    return any(h in toks for h in HEADWORDS)

def dyn_cutoff(ngram_len: int, char_len: int) -> int:
    # Disable short, low-information matches
    if ngram_len <= 1 or char_len < 6:
        return 101  # effectively disable
    if ngram_len == 2:
        return 92
    return 88  # 3+ tokens

def in_negated_scope(text: str, start: int) -> bool:
    # look back ~80 chars or until punctuation
    window = text[max(0, start-80):start].lower()
    # stop at latest clause boundary
    m = re.search(r"([.;:!?]|,)\s*([^.;:!?]+)$", window)
    if m:
        window = m.group(2)
    if any(re.search(rf"\b{re.escape(trig)}\b", window) for trig in NEG_TRIGGERS):
        return True
    return False

def in_uncertain_scope(text: str, start: int) -> bool:
    window = text[max(0, start-80):start].lower()
    m = re.search(r"([.;:!?]|,)\s*([^.;:!?]+)$", window)
    if m:
        window = m.group(2)
    return any(u in window for u in UNCERTAIN)

def extract_character_spans(
    text: str, symptom_list: List[str], synonym_map: Dict[str, str]
) -> List[Tuple[int, int, str, float]]:
    """Exact matching with boundary checks; returns (start, end, canonical_label, confidence)."""
    if not text:
        return []
    all_phrases = build_phrase_map(symptom_list, synonym_map)
    phrases = sorted(all_phrases.items(), key=lambda x: len(x[0]), reverse=True)
    spans: List[Tuple[int, int, str, float]] = []
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
            left_ok = (pos == 0) or (not text[pos - 1].isalnum())
            right_ok = (end_pos == len(text)) or (not text[end_pos].isalnum())
            if left_ok and right_ok:
                rng = set(range(pos, end_pos))
                if not rng.intersection(used):
                    # Scope filters
                    if in_negated_scope(text, pos) or in_uncertain_scope(text, pos):
                        start = pos + 1
                        continue
                    spans.append((pos, end_pos, all_phrases[phrase], 1.0))
                    used.update(rng)
            start = pos + 1

    spans.sort(key=lambda x: x[0])
    return spans

def fuzzy_match_one(query: str, choices: List[str], score_cutoff: int) -> Optional[str]:
    """Best WRatio match if score >= cutoff; else None."""
    if not query or not choices:
        return None
    result = process.extractOne(query, choices, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
    if result:
        match, score, _ = result
        return match
    return None

def generate_token_spans(text: str) -> List[Tuple[str, int, int]]:
    """Tokenize to (token_lower, start, end), alnum-only tokens."""
    tokens: List[Tuple[str, int, int]] = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isalnum():
            j = i
            while j < n and text[j].isalnum():
                j += 1
            tokens.append((text[i:j].lower(), i, j))
            i = j
        else:
            i += 1
    return tokens

def extract_fuzzy_spans(
    text: str,
    symptom_list: List[str],
    synonym_map: Dict[str, str],
    score_cutoff: int = 85,
    max_ngram: int = 3,
) -> Tuple[List[Tuple[int, int, str, float]], List[Tuple[str, int]]]:
    """
    Headword-gated fuzzy matching over 2..max_ngram token n-grams; returns:
    - spans: list of (start, end, canonical_label, confidence)
    - suggestions: list of (canonical_label, score)
    """
    if not text:
        return [], []

    all_phrases = build_phrase_map(symptom_list, synonym_map)
    phrase_keys = list(all_phrases.keys())
    tokens = generate_token_spans(text)
    if not tokens:
        return [], []

    used = set()
    spans: List[Tuple[int, int, str, float]] = []
    best_candidates: Dict[str, int] = {}

    # Search longer n-grams first; disable 1-grams
    for ngram in range(min(max_ngram, len(tokens)), 1, -1):
        for i in range(0, len(tokens) - ngram + 1):
            start_tok = tokens[i][1]
            end_tok = tokens[i + ngram - 1][2]
            query = text[start_tok:end_tok].lower()
            if len(query) < 6:
                continue
            # Headword gating
            if not contains_headword(query):
                continue
            cut = max(score_cutoff, dyn_cutoff(ngram, len(query)))
            if cut >= 100:
                continue
            matched_key = fuzzy_match_one(query, phrase_keys, score_cutoff=cut)
            if not matched_key:
                continue
            # word boundary check
            left_ok = (start_tok == 0) or (not text[start_tok - 1].isalnum())
            right_ok = (end_tok == len(text)) or (not text[end_tok].isalnum())
            if not (left_ok and right_ok):
                continue
            rng = set(range(start_tok, end_tok))
            if rng.intersection(used):
                continue
            if in_negated_scope(text, start_tok) or in_uncertain_scope(text, start_tok):
                continue
            canon_label = all_phrases[matched_key]
            conf = 0.9 if ngram >= 3 else 0.85
            spans.append((start_tok, end_tok, canon_label, conf))
            used.update(rng)
            best_candidates[canon_label] = max(best_candidates.get(canon_label, 0), 100)

    # Suggestions for mining if no spans
    if not spans:
        probe_strings = []
        for ngram in range(min(max_ngram, len(tokens)), 1, -1):
            for i in range(0, len(tokens) - ngram + 1):
                s, e = tokens[i][1], tokens[i + ngram - 1][2]
                q = text[s:e].lower()
                if len(q) >= 6 and contains_headword(q):
                    probe_strings.append(q)
                if len(probe_strings) >= 20:
                    break
            if len(probe_strings) >= 20:
                break
        for q in probe_strings:
            res = process.extractOne(q, phrase_keys, scorer=fuzz.WRatio)
            if res:
                key, score, _ = res
                canon_label = all_phrases[key]
                best_candidates[canon_label] = max(best_candidates.get(canon_label, 0), int(score))

    suggestions = sorted(best_candidates.items(), key=lambda x: x[1], reverse=True)[:5]
    spans.sort(key=lambda x: x[0])
    return spans, suggestions

def auto_label_sentences(
    csv_file: str,
    data_dir: str = DATA_DIR_DEFAULT,
    max_sentences: int = None,
    fuzzy_cutoff: int = 90,
    fuzzy_max_ngram: int = 4,
) -> Tuple[str, str]:
    """
    Process CSV and write:
    - data/auto_labeled_sample.jsonl (ONLY sentences with spans)
    - data/unknown_spans.jsonl (sentences with no spans, plus suggestions)
    """
    symptoms, synonyms = load_symptom_data(data_dir)
    synonyms = sanitize_synonyms(synonyms)

    labeled_file = os.path.join(data_dir, "auto_labeled_sample.jsonl")
    unknown_file = os.path.join(data_dir, "unknown_spans.jsonl")

    chunk_size = 1000
    kept = 0
    labeled: List[dict] = []
    unknown: List[dict] = []

    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        # comment out max_sentences check to auto-label full csv
        # if kept >= max_sentences:
        #     break
        for _, row in chunk.iterrows():
            if kept >= max_sentences:
                break
            # Restrict scope: Description and Patient only
            for col in ["Description", "Patient"]:
                if col in row and pd.notna(row[col]):
                    text = str(row[col]).strip()
                    if not text or len(text) <= 10:
                        continue
                    # 1) exact
                    spans = extract_character_spans(text, symptoms, synonyms)
                    # 2) fuzzy fallback when no exact matches
                    if not spans:
                        spans, suggestions = extract_fuzzy_spans(
                            text,
                            symptoms,
                            synonyms,
                            score_cutoff=fuzzy_cutoff,
                            max_ngram=fuzzy_max_ngram,
                        )
                    else:
                        suggestions = []
                    if spans:
                        labeled.append(
                            {
                                "text": text,
                                "spans": spans,  # (start, end, canonical, confidence)
                                "source_column": col,
                                "row_index": int(getattr(row, "name", kept)),
                            }
                        )
                        kept += 1
                        if kept >= max_sentences:
                            break
                    else:
                        unknown.append(
                            {
                                "text": text,
                                "source_column": col,
                                "row_index": int(getattr(row, "name", len(unknown))),
                                "suggestions": suggestions,  # list of [label, score]
                            }
                        )

    os.makedirs(data_dir, exist_ok=True)
    with open(labeled_file, "w", encoding="utf-8") as f:
        for entry in labeled:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    with open(unknown_file, "w", encoding="utf-8") as f:
        for entry in unknown:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    total_spans = sum(len(e["spans"]) for e in labeled)
    print(f"Kept sentences: {len(labeled)} | Total spans: {total_spans} | Unknown sentences: {len(unknown)}")
    return labeled_file, unknown_file

def validate_spans(jsonl_file: str, num_examples: int = 5) -> None:
    """Print sample lines with inline highlighting and span tuples (start,end,label,conf)."""
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
            highlighted = text
            # Replace from right to left
            for start, end, label, conf in sorted(spans, key=lambda x: x[0], reverse=True):
                highlighted = highlighted[:start] + f"[{text[start:end]}]({label}|{conf:.2f})" + highlighted[end:]
            print(f"Highlighted: {highlighted}")
            shown += 1

def main():
    parser = argparse.ArgumentParser(
        description="Auto-label medical symptoms with exact + gated fuzzy; writes positives and negatives."
    )
    parser.add_argument("-i","--input", default=DEFAULT_INPUT, help="Input CSV (default: data/ai-medical-chatbot.csv)")
    parser.add_argument("-d","--data-dir", default=DATA_DIR_DEFAULT, help="Data dir with JSON (default: data)")
    parser.add_argument("-m","--max-sentences", type=int, default=200, help="Max kept sentences with spans (default: 200)")
    parser.add_argument("--fuzzy-cutoff", type=int, default=90, help="WRatio cutoff (default: 90)")
    parser.add_argument("--fuzzy-max-ngram", type=int, default=4, help="Max n-gram length (default: 4)")
    parser.add_argument("-v","--validate", action="store_true", help="Print validation examples")
    parser.add_argument("-e","--examples", type=int, default=3, help="Number of examples to print")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data dir not found: {args.data_dir}")
    for name in ["symptoms.json", "synonyms.json"]:
        if not os.path.exists(os.path.join(args.data_dir, name)):
            raise FileNotFoundError(f"Missing {name} in {args.data_dir}")

    print("Running auto-label...")
    labeled_path, unknown_path = auto_label_sentences(
        args.input,
        data_dir=args.data_dir,
        max_sentences=args.max_sentences,
        fuzzy_cutoff=args.fuzzy_cutoff,
        fuzzy_max_ngram=args.fuzzy_max_ngram,
    )
    if args.validate:
        validate_spans(labeled_path, num_examples=args.examples)
    print(f"Saved labeled: {labeled_path}")
    print(f"Saved unknown: {unknown_path}")

if __name__ == "__main__":
    main()
