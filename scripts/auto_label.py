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
from typing import List, Dict, Tuple, Optional, Set
import pandas as pd
from collections import Counter, defaultdict
import csv
import random
import io


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Prefer project helpers if available; otherwise provide fallbacks
try:
    from chatbot.symptom_extractor import load_symptom_data, canonicalize_symptom
except Exception:
    def canonicalize_symptom(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    def load_symptom_data(data_dir: str):
        """Fallback: Load symptoms from CSV (preferred) or JSON, and synonyms from JSON."""
        symptoms = []
        symptoms_csv_path = os.path.join(data_dir, "symptoms.csv")
        symptoms_json_path = os.path.join(data_dir, "symptoms.json")
        
        # Try CSV first
        if os.path.exists(symptoms_csv_path):
            import csv
            with open(symptoms_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symptom = row["symptoms"].strip()
                    if symptom:
                        symptoms.append(symptom)
        # Fallback to JSON
        elif os.path.exists(symptoms_json_path):
            with open(symptoms_json_path, "r", encoding="utf-8") as f:
                symptoms = json.load(f)
        else:
            raise FileNotFoundError(f"Neither symptoms.csv nor symptoms.json found in {data_dir}")
        
        # Load synonyms
        synonyms_path = os.path.join(data_dir, "synonyms.json")
        if os.path.exists(synonyms_path):
            with open(synonyms_path, "r", encoding="utf-8") as f:
                synonyms = json.load(f)
        else:
            synonyms = {}
        
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

def _load_json_relaxed(path: str) -> Dict[str, str]:
    """Load JSON and auto-fix common format issues (like trailing commas).
    If fixing is needed, the corrected JSON is written back to the same path.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Remove trailing commas before } or ]
        fixed = re.sub(r",\s*([}\]])", r"\1", raw)
        # Collapse duplicate commas
        fixed = re.sub(r",\s*,+", ",", fixed)
        try:
            data = json.loads(fixed)
            # Write back the fixed JSON for future runs
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return data
        except json.JSONDecodeError:
            # If still invalid, fall back to empty map but warn
            print(f"Warning: Could not parse JSON in {path}; proceeding with empty synonyms.")
            return {}

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

def append_tail_filler(text: str) -> str:
    """Append small filler at the end to preserve existing character span indices."""
    fillers = [
        "to be honest.",
        "if that makes sense.",
        "you know.",
        "honestly.",
        "kind of.",
        "a bit.",
        "for what it's worth.",
        "these days.",
        "lately.",
        "right now.",
    ]
    # 70% chance to append a filler
    if random.random() < 0.7:
        return text.rstrip() + " " + random.choice(fillers)
    return text

def balance_labeled_sentences(
    labeled: List[dict],
    max_per_class: int = 1000,
    min_per_class: int = 900,
    random_seed: Optional[int] = 42,
    use_augmentation: bool = True,
) -> List[dict]:
    """
    Balance labeled sentences by downsampling frequent classes and upsampling rare classes.
    
    Args:
        labeled: List of labeled sentence dictionaries with 'spans' field
        max_per_class: Maximum number of sentences per label (downsample cap)
        min_per_class: Minimum number of sentences per label (upsample target)
        random_seed: Random seed for reproducibility
    
    Returns:
        Balanced list of labeled sentences (may contain duplicates for upsampled classes)
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    if not labeled:
        return labeled
    
    # Build label -> sentence indices mapping
    label_to_sentences: Dict[str, List[int]] = defaultdict(list)
    
    for idx, entry in enumerate(labeled):
        spans = entry.get("spans", [])
        # Extract unique labels from this sentence
        labels_in_sentence = {lab for (_, _, lab, _) in spans}
        for label in labels_in_sentence:
            label_to_sentences[label].append(idx)
    
    # Start with all unique indices seen
    kept_indices: Set[int] = set()
    augmented_entries: List[dict] = []
    
    for label, sentence_indices in sorted(label_to_sentences.items()):
        count = len(sentence_indices)
        if count > max_per_class:
            # Downsample
            sampled = set(random.sample(sentence_indices, max_per_class))
            kept_indices.update(sampled)
        else:
            kept_indices.update(sentence_indices)
            if count < min_per_class and use_augmentation and sentence_indices:
                need = min_per_class - count
                # Create augmented copies to reach at least min_per_class
                picks = random.choices(sentence_indices, k=need)
                for idx in picks:
                    base = labeled[idx]
                    new_entry = {
                        "text": append_tail_filler(base["text"]),
                        "spans": base.get("spans", []),
                        "source_column": base.get("source_column"),
                        "row_index": base.get("row_index", idx),
                        "augmented": True,
                        "aug_label": label,
                    }
                    augmented_entries.append(new_entry)
    
    balanced = [labeled[idx] for idx in sorted(kept_indices)] + augmented_entries
    
    # Update counts after balancing
    balanced_counts = Counter()
    for entry in balanced:
        spans = entry.get("spans", [])
        labels_in_sentence = {lab for (_, _, lab, _) in spans}
        for label in labels_in_sentence:
            balanced_counts[label] += 1
    
    print(f"Balancing: {len(labeled)} -> {len(balanced)} sentences")
    print(f"Label distribution after balancing (target: {min_per_class}-{max_per_class} per label):")
    for label, count in sorted(balanced_counts.items(), key=lambda x: -x[1])[:10]:
        status = "✓" if min_per_class <= count <= max_per_class else "⚠"
        print(f"  {status} {label}: {count} sentences")
    
    return balanced

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
    # edit max_sentences for total number of (positive) sentences for BIO tagging
    max_sentences: int = 2000,
    fuzzy_cutoff: int = 90,
    fuzzy_max_ngram: int = 4,
) -> Tuple[str, str]:
    """
    Process CSV and write:
    - data/auto_labeled_sample.jsonl (ONLY sentences with spans)
    - data/unknown_spans.jsonl (sentences with no spans, plus suggestions)
    - data/label_counts.csv (per-label span and sentence counts)
    """
    symptoms, synonyms = load_symptom_data(data_dir)
    synonyms = sanitize_synonyms(synonyms)

    labeled_file = os.path.join(data_dir, "auto_labeled_sample.jsonl")
    unknown_file = os.path.join(data_dir, "unknown_spans.jsonl")
    counts_file = os.path.join(data_dir, "label_counts.csv")

    os.makedirs(data_dir, exist_ok=True)

    chunk_size = 1000
    kept = 0
    labeled: List[dict] = []
    unknown: List[dict] = []

    # NEW: per-label counts
    span_counts: Counter = Counter()
    sentence_counts: Counter = Counter()

    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        if max_sentences is not None and kept >= max_sentences: break  # keep disabled to label full CSV

        for _, row in chunk.iterrows():
            if max_sentences is not None and kept >= max_sentences: break  # keep disabled to label full CSV

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
                        labeled.append({
                            "text": text,
                            "spans": spans,  # (start, end, canonical, confidence)
                            "source_column": col,
                            "row_index": int(getattr(row, "name", kept)),
                        })
                        kept += 1

                        # NEW: update counts
                        labels_in_sentence = {lab for (_, _, lab, _) in spans}
                        for lab in labels_in_sentence:
                            sentence_counts[lab] += 1
                        for (_, _, lab, _) in spans:
                            span_counts[lab] += 1
                    else:
                        unknown.append({
                            "text": text,
                            "source_column": col,
                            "row_index": int(getattr(row, "name", len(unknown))),
                            "suggestions": suggestions,  # list of [label, score]
                        })

    # Balance labeled sentences: downsample frequent classes, upsample rare classes
    labeled = balance_labeled_sentences(
        labeled,
        max_per_class=1000,  # Cap frequent classes like fever at 1000 instances
        min_per_class=900,   # Upsample rare symptoms to at least 900 instances
        random_seed=42,
    )

    # Recompute counts after balancing
    span_counts = Counter()
    sentence_counts = Counter()
    for entry in labeled:
        spans = entry.get("spans", [])
        labels_in_sentence = {lab for (_, _, lab, _) in spans}
        for lab in labels_in_sentence:
            sentence_counts[lab] += 1
        for (_, _, lab, _) in spans:
            span_counts[lab] += 1

    # Write outputs
    with open(labeled_file, "w", encoding="utf-8") as f:
        for entry in labeled:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    with open(unknown_file, "w", encoding="utf-8") as f:
        for entry in unknown:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    # NEW: write per-label counts CSV (after balancing)
    with open(counts_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "span_count", "sentence_count"])
        # sort by span_count desc, then sentence_count desc, then label
        for lab in sorted(span_counts.keys(), key=lambda k: (-span_counts[k], -sentence_counts[k], k)):
            writer.writerow([lab, span_counts[lab], sentence_counts[lab]])

    total_spans = sum(len(e["spans"]) for e in labeled)
    covered_labels = set(span_counts.keys())
    total_labels = len(symptoms)
    print(f"Kept sentences: {len(labeled)} | Total spans: {total_spans} | Unknown sentences: {len(unknown)}")
    print(f"Label coverage: {len(covered_labels)}/{total_labels} canonical symptoms matched at least once")
    print(f"Wrote per-label counts to: {counts_file}")

    return labeled_file, unknown_file


def _primary_label_for_sentence(spans: List[Tuple[int, int, str, float]], label_order: List[str]) -> Optional[str]:
    """Pick a primary label for multi-label sentences to simplify allocation.
    Prefer the label with highest global order (frequency-based order provided)."""
    if not spans:
        return None
    labels = {lab for (_, _, lab, _) in spans}
    order_index = {lab: i for i, lab in enumerate(label_order)}
    # smallest index means higher priority
    chosen = sorted(labels, key=lambda x: order_index.get(x, 10**9))
    return chosen[0] if chosen else None


def generate_silver_and_gold(
    csv_file: str,
    data_dir: str = DATA_DIR_DEFAULT,
    silver_pos_target: int = 50000,
    silver_neg_target: int = 50000,
    gold_pos_target: int = 5000,
    gold_neg_target: int = 5000,
    max_per_class: int = 1000,
    min_per_class: int = 900,
    fuzzy_cutoff: int = 90,
    fuzzy_max_ngram: int = 4,
) -> Tuple[str, str, str]:
    """
    Build silver set (50k pos/50k neg) with per-symptom 900-1000 targets (subject to total size),
    and gold validation set (5k pos/5k neg) mirroring silver distribution.
    """
    symptoms, synonyms = load_symptom_data(data_dir)
    synonyms = sanitize_synonyms(synonyms)

    silver_file = os.path.join(data_dir, "auto_labeled.jsonl")
    gold_file = os.path.join(data_dir, "validation.jsonl")
    counts_file = os.path.join(data_dir, "label_counts.csv")

    pos_entries: List[dict] = []
    neg_entries: List[dict] = []

    # Extract sentences until we hit silver_pos_target positives
    for chunk in pd.read_csv(csv_file, chunksize=1000):
        if len(pos_entries) >= silver_pos_target:
            break
        for _, row in chunk.iterrows():
            if len(pos_entries) >= silver_pos_target:
                break
            for col in ["Description", "Patient"]:
                if col in row and pd.notna(row[col]):
                    text = str(row[col]).strip()
                    if not text or len(text) <= 10:
                        continue
                    spans = extract_character_spans(text, symptoms, synonyms)
                    suggestions: List[Tuple[str, int]] = []
                    if not spans:
                        spans, suggestions = extract_fuzzy_spans(
                            text,
                            symptoms,
                            synonyms,
                            score_cutoff=fuzzy_cutoff,
                            max_ngram=fuzzy_max_ngram,
                        )
                    entry = {
                        "text": text,
                        "spans": spans,
                        "source_column": col,
                        "row_index": int(getattr(row, "name", len(pos_entries)+len(neg_entries))),
                    }
                    if spans:
                        pos_entries.append(entry)
                    else:
                        neg_entries.append(entry)

    # Compute initial per-label counts (sentence-level)
    sentence_counts = Counter()
    for e in pos_entries:
        labels = {lab for (_, _, lab, _) in e.get("spans", [])}
        for lab in labels:
            sentence_counts[lab] += 1

    # Determine label allocation within silver_pos_target budget
    labels_sorted = [lab for lab, _ in sorted(sentence_counts.items(), key=lambda x: -x[1])]
    allocation: Dict[str, int] = {}
    remaining = silver_pos_target
    for lab in labels_sorted:
        base = sentence_counts[lab]
        target = max(min(base, max_per_class), min_per_class) if base > 0 else 0
        if target <= 0:
            continue
        take = min(target, remaining)
        allocation[lab] = take
        remaining -= take
        if remaining <= 0:
            break

    # Bucket sentences by a primary label using the frequency order
    buckets: Dict[str, List[int]] = defaultdict(list)
    order = labels_sorted
    for idx, e in enumerate(pos_entries):
        plab = _primary_label_for_sentence(e.get("spans", []), order)
        if plab is not None and plab in allocation:
            buckets[plab].append(idx)

    # Build balanced silver positives per allocation; augment tail to upsample
    silver_pos: List[dict] = []
    for lab, need in allocation.items():
        pool = buckets.get(lab, [])
        if not pool:
            continue
        if len(pool) >= need:
            chosen = random.sample(pool, need)
            silver_pos.extend(pos_entries[i] for i in chosen)
        else:
            # take all, then augment to reach need
            silver_pos.extend(pos_entries[i] for i in pool)
            deficit = need - len(pool)
            picks = random.choices(pool, k=deficit)
            for i in picks:
                base = pos_entries[i]
                silver_pos.append({
                    "text": append_tail_filler(base["text"]),
                    "spans": base.get("spans", []),
                    "source_column": base.get("source_column"),
                    "row_index": base.get("row_index", i),
                    "augmented": True,
                    "aug_label": lab,
                })

    # If still under target due to rounding/allocation gaps, top up from remaining frequent pools
    if len(silver_pos) < silver_pos_target:
        deficit = silver_pos_target - len(silver_pos)
        flat_pool = [i for lab in allocation for i in buckets.get(lab, [])]
        if flat_pool:
            picks = random.choices(flat_pool, k=deficit)
            for i in picks:
                base = pos_entries[i]
                silver_pos.append({
                    "text": append_tail_filler(base["text"]),
                    "spans": base.get("spans", []),
                    "source_column": base.get("source_column"),
                    "row_index": base.get("row_index", i),
                    "augmented": True,
                })
    elif len(silver_pos) > silver_pos_target:
        silver_pos = random.sample(silver_pos, silver_pos_target)

    # Silver negatives: fill to target with replacement if needed
    if len(neg_entries) < silver_neg_target:
        deficit = silver_neg_target - len(neg_entries)
        if neg_entries:
            neg_entries.extend(random.choices(neg_entries, k=deficit))
    elif len(neg_entries) > silver_neg_target:
        neg_entries = random.sample(neg_entries, silver_neg_target)

    # Recompute counts from silver_pos for output
    span_counts = Counter()
    sent_counts = Counter()
    for e in silver_pos:
        labels = {lab for (_, _, lab, _) in e.get("spans", [])}
        for lab in labels:
            sent_counts[lab] += 1
        for (_, _, lab, _) in e.get("spans", []):
            span_counts[lab] += 1

    # Write silver set
    with open(silver_file, "w", encoding="utf-8") as f:
        for e in silver_pos:
            json.dump(e, f, ensure_ascii=False)
            f.write("\n")
        for e in neg_entries:
            # Ensure negatives have empty spans
            out = dict(e)
            out["spans"] = []
            json.dump(out, f, ensure_ascii=False)
            f.write("\n")

    # Write counts
    with open(counts_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "span_count", "sentence_count"])
        for lab in sorted(span_counts.keys(), key=lambda k: (-span_counts[k], -sent_counts[k], k)):
            writer.writerow([lab, span_counts[lab], sent_counts[lab]])

    # Build gold validation mirroring silver distribution (positives)
    # Derive per-label proportions from silver_pos (sentence-level)
    total_pos = len(silver_pos)
    gold_allocation: Dict[str, int] = {}
    for lab, cnt in sent_counts.items():
        take = max(1, int(round(cnt / total_pos * gold_pos_target)))
        gold_allocation[lab] = take
    # Sample positives per label from silver_pos
    silver_pos_by_label: Dict[str, List[int]] = defaultdict(list)
    silver_pos_labels_order = [lab for lab, _ in sorted(sent_counts.items(), key=lambda x: -x[1])]
    for idx, e in enumerate(silver_pos):
        plab = _primary_label_for_sentence(e.get("spans", []), silver_pos_labels_order)
        if plab is not None:
            silver_pos_by_label[plab].append(idx)
    gold_pos: List[dict] = []
    for lab, need in gold_allocation.items():
        pool = silver_pos_by_label.get(lab, [])
        if pool:
            if len(pool) >= need:
                gold_pos.extend(silver_pos[i] for i in random.sample(pool, need))
            else:
                gold_pos.extend(silver_pos[i] for i in pool)
                extra = need - len(pool)
                gold_pos.extend(silver_pos[i] for i in random.choices(pool, k=extra))
    if len(gold_pos) > gold_pos_target:
        gold_pos = random.sample(gold_pos, gold_pos_target)
    elif len(gold_pos) < gold_pos_target and gold_pos:
        gold_pos.extend(random.choices(gold_pos, k=gold_pos_target - len(gold_pos)))

    # Gold negatives: sample from silver negatives to 5k
    gold_neg = random.sample(neg_entries, min(len(neg_entries), gold_neg_target)) if neg_entries else []
    if len(gold_neg) < gold_neg_target and gold_neg:
        gold_neg.extend(random.choices(gold_neg, k=gold_neg_target - len(gold_neg)))

    # Write gold set
    with open(gold_file, "w", encoding="utf-8") as f:
        for e in gold_pos:
            json.dump(e, f, ensure_ascii=False)
            f.write("\n")
        for e in gold_neg:
            out = dict(e)
            out["spans"] = []
            json.dump(out, f, ensure_ascii=False)
            f.write("\n")

    print(f"Silver set -> {silver_file}: {len(silver_pos)} positives, {len(neg_entries)} negatives")
    print(f"Gold set   -> {gold_file}: {len(gold_pos)} positives, {len(gold_neg)} negatives")
    print(f"Counts     -> {counts_file}")
    return silver_file, counts_file, gold_file

def oversample_low_count_sentences(
    data_dir: str = DATA_DIR_DEFAULT,
    auto_labeled_filename: str = "auto_labeled.jsonl",
    counts_filename: str = "label_counts.csv",
    min_per_class: int = 900,
    max_per_class: int = 1000,
    random_seed: Optional[int] = 42,
) -> Tuple[str, str]:
    """
    Oversample positive sentences for symptoms with low counts in data/auto_labeled.jsonl
    using safe tail-only grammar/phrase perturbations, then update label_counts.csv.

    - Only positives (entries with non-empty spans) are considered for symptom counts.
    - For each label with sentence_count < min_per_class, duplicate sentences with that label
      using light tail-only fillers to avoid breaking character offsets.
    - Does not downsample labels > max_per_class (non-destructive).
    """
    if random_seed is not None:
        random.seed(random_seed)

    auto_path = os.path.join(data_dir, auto_labeled_filename)
    counts_path = os.path.join(data_dir, counts_filename)

    # Read existing dataset
    entries: List[dict] = []
    with open(auto_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    # Split positives/negatives
    pos_indices: List[int] = []
    neg_indices: List[int] = []
    for i, e in enumerate(entries):
        spans = e.get("spans", [])
        if spans:
            pos_indices.append(i)
        else:
            neg_indices.append(i)

    # Compute sentence counts per label on positives
    sentence_counts: Counter = Counter()
    for i in pos_indices:
        e = entries[i]
        labels = {lab for (_, _, lab, _) in e.get("spans", [])}
        for lab in labels:
            sentence_counts[lab] += 1

    # Map label -> indices of positive entries containing that label
    label_to_pos_indices: Dict[str, List[int]] = defaultdict(list)
    for i in pos_indices:
        e = entries[i]
        labels = {lab for (_, _, lab, _) in e.get("spans", [])}
        for lab in labels:
            label_to_pos_indices[lab].append(i)

    # Augment to lift underrepresented labels up to min_per_class (capped by max_per_class)
    augmented: List[dict] = []
    for lab, count in sentence_counts.items():
        target = min_per_class
        cap = max_per_class
        if count >= target:
            continue
        pool = label_to_pos_indices.get(lab, [])
        if not pool:
            continue
        need = min(cap, target) - count
        if need <= 0:
            continue
        picks = random.choices(pool, k=need)
        for idx in picks:
            base = entries[idx]
            augmented.append({
                "text": append_tail_filler(base.get("text", "")),
                "spans": base.get("spans", []),
                "source_column": base.get("source_column"),
                "row_index": base.get("row_index", idx),
                "augmented": True,
                "aug_label": lab,
            })

    if augmented:
        entries.extend(augmented)

    # Recompute counts (span and sentence) on positives only
    span_counts: Counter = Counter()
    sent_counts: Counter = Counter()
    for e in entries:
        spans = e.get("spans", [])
        if not spans:
            continue
        labels = {lab for (_, _, lab, _) in spans}
        for lab in labels:
            sent_counts[lab] += 1
        for (_, _, lab, _) in spans:
            span_counts[lab] += 1

    # Write updated dataset back
    with open(auto_path, "w", encoding="utf-8") as f:
        for e in entries:
            json.dump(e, f, ensure_ascii=False)
            f.write("\n")

    # Write updated counts
    with open(counts_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "span_count", "sentence_count"])
        for lab in sorted(span_counts.keys(), key=lambda k: (-span_counts[k], -sent_counts[k], k)):
            writer.writerow([lab, span_counts[lab], sent_counts[lab]])

    print(f"Oversampled {len(augmented)} new positive entries for low-count symptoms.")
    print(f"Updated dataset: {auto_path}")
    print(f"Updated counts:  {counts_path}")
    return auto_path, counts_path

def downsample_high_count_sentences(
    data_dir: str = DATA_DIR_DEFAULT,
    auto_labeled_filename: str = "auto_labeled.jsonl",
    counts_filename: str = "label_counts.csv",
    max_per_class: int = 1000,
    random_seed: Optional[int] = 42,
) -> Tuple[str, str]:
    """
    Downsample positives in data/auto_labeled.jsonl so that per-symptom sentence counts
    do not exceed max_per_class. Negatives are preserved as-is.

    Strategy:
      1) Primary-label bucketing by frequency order and cap each bucket to max_per_class.
      2) Second pass: enforce caps for any label still exceeding the threshold by removing
         additional positives that contain the label (randomly), until all are <= max.
    """
    if random_seed is not None:
        random.seed(random_seed)

    auto_path = os.path.join(data_dir, auto_labeled_filename)
    counts_path = os.path.join(data_dir, counts_filename)

    # Read dataset
    entries: List[dict] = []
    with open(auto_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    # Split indices
    pos_indices: List[int] = []
    neg_indices: List[int] = []
    for i, e in enumerate(entries):
        if e.get("spans", []):
            pos_indices.append(i)
        else:
            neg_indices.append(i)

    # Initial sentence counts
    sentence_counts = Counter()
    for i in pos_indices:
        labs = {lab for (_, _, lab, _) in entries[i].get("spans", [])}
        for lab in labs:
            sentence_counts[lab] += 1

    # Buckets by primary label
    label_order = [lab for lab, _ in sorted(sentence_counts.items(), key=lambda x: -x[1])]
    buckets: Dict[str, List[int]] = defaultdict(list)
    for i in pos_indices:
        plab = _primary_label_for_sentence(entries[i].get("spans", []), label_order)
        if plab is not None:
            buckets[plab].append(i)

    # First pass: cap by bucket
    keep_set: Set[int] = set()
    for lab, idxs in buckets.items():
        if len(idxs) <= max_per_class:
            keep_set.update(idxs)
        else:
            keep_set.update(random.sample(idxs, max_per_class))

    # Add negatives
    keep_set.update(neg_indices)

    # Provisional filtered entries
    filtered_entries = [e for j, e in enumerate(entries) if j in keep_set]

    # Second pass: enforce caps globally
    # Build mapping from label to indices in filtered_entries
    filt_pos_indices: List[int] = []
    label_to_indices: Dict[str, List[int]] = defaultdict(list)
    for j, e in enumerate(filtered_entries):
        spans = e.get("spans", [])
        if not spans:
            continue
        filt_pos_indices.append(j)
        labs = {lab for (_, _, lab, _) in spans}
        for lab in labs:
            label_to_indices[lab].append(j)

    # Compute counts
    filt_counts = Counter()
    for lab, idxs in label_to_indices.items():
        filt_counts[lab] = len(idxs)

    # Determine drops needed per label
    drop_set: Set[int] = set()
    for lab, cnt in sorted(filt_counts.items(), key=lambda x: -x[1]):
        if cnt <= max_per_class:
            continue
        excess = cnt - max_per_class
        pool = [j for j in label_to_indices.get(lab, []) if j not in drop_set]
        if pool:
            drop = set(random.sample(pool, min(excess, len(pool))))
            drop_set.update(drop)

    if drop_set:
        final_entries = [e for j, e in enumerate(filtered_entries) if j not in drop_set]
    else:
        final_entries = filtered_entries

    # Recompute counts for output (positives only)
    span_counts = Counter()
    sent_counts = Counter()
    for e in final_entries:
        spans = e.get("spans", [])
        if not spans:
            continue
        labs = {lab for (_, _, lab, _) in spans}
        for lab in labs:
            sent_counts[lab] += 1
        for (_, _, lab, _) in spans:
            span_counts[lab] += 1

    # Write updated dataset
    with open(auto_path, "w", encoding="utf-8") as f:
        for e in final_entries:
            json.dump(e, f, ensure_ascii=False)
            f.write("\n")

    # Write updated counts
    with open(counts_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "span_count", "sentence_count"])
        for lab in sorted(span_counts.keys(), key=lambda k: (-span_counts[k], -sent_counts[k], k)):
            writer.writerow([lab, span_counts[lab], sent_counts[lab]])

    print(f"Downsampled positives to enforce <= {max_per_class} per symptom.")
    print(f"Updated dataset: {auto_path}")
    print(f"Updated counts:  {counts_path}")
    return auto_path, counts_path

def deduplicate_symptoms_and_update_synonyms(
    data_dir: str = DATA_DIR_DEFAULT,
    csv_filename: str = "symptoms.csv",
    json_filename: str = "symptoms.json",
    synonyms_filename: str = "synonyms.json",
    similarity_threshold: int = 90,
) -> Tuple[str, str, str]:
    """
    Go through each symptom in symptoms.csv, map near-duplicates to an existing canonical symptom
    if similarity >= threshold, update synonyms.json accordingly, and write de-duplicated
    canonical list back to symptoms.csv and symptoms.json.
    """
    csv_path = os.path.join(data_dir, csv_filename)
    json_path = os.path.join(data_dir, json_filename)
    syn_path = os.path.join(data_dir, synonyms_filename)

    # Load current symptoms (CSV primary) and existing synonyms
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_filename} in {data_dir}")
    
    # Use csv module to handle commas within symptom names properly
    raw_symptoms: List[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "symptoms" not in reader.fieldnames:
            raise ValueError("symptoms.csv must contain a 'symptoms' column")
        for row in reader:
            symptom = row["symptoms"].strip() if row["symptoms"] else ""
            if symptom:
                raw_symptoms.append(symptom)

    if os.path.exists(syn_path):
        synonym_map: Dict[str, str] = _load_json_relaxed(syn_path)
    else:
        synonym_map = {}

    canonical_list: List[str] = []
    canonical_keys: List[str] = []  # normalized for matching

    for phrase in raw_symptoms:
        norm = canonicalize_symptom(phrase)
        if not canonical_list:
            canonical_list.append(phrase)
            canonical_keys.append(norm)
            continue
        # Find best match among canonical keys
        result = process.extractOne(norm, canonical_keys, scorer=fuzz.WRatio)
        if result is None:
            # No canonical yet
            canonical_list.append(phrase)
            canonical_keys.append(norm)
            continue
        best_key, score, idx = result
        if score >= similarity_threshold:
            # Treat as synonym of existing canonical
            canonical = canonical_list[idx]
            if phrase != canonical:
                synonym_map[phrase] = canonical
        else:
            # Add as new canonical
            canonical_list.append(phrase)
            canonical_keys.append(norm)

    # Sort canonicals for stability
    canonical_list_sorted = sorted(set(canonical_list), key=lambda s: canonicalize_symptom(s))

    # Write updated symptoms.csv (quote fields with commas)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symptoms"])
        for symptom in canonical_list_sorted:
            writer.writerow([symptom])

    # Write updated symptoms.json (mirror CSV)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(canonical_list_sorted, f, ensure_ascii=False, indent=2)

    # Write updated synonyms.json
    with open(syn_path, "w", encoding="utf-8") as f:
        json.dump(synonym_map, f, ensure_ascii=False, indent=2)

    print(f"Deduplicated symptoms -> {csv_path} and {json_path}")
    print(f"Updated synonyms     -> {syn_path}")
    return csv_path, json_path, syn_path

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
    # New dataset generation flags
    parser.add_argument("--make-datasets", action="store_true", help="Generate silver (50k/50k) and gold (5k/5k) sets")
    parser.add_argument("--silver-pos", type=int, default=50000, help="Silver positives target (default: 50000)")
    parser.add_argument("--silver-neg", type=int, default=50000, help="Silver negatives target (default: 50000)")
    parser.add_argument("--gold-pos", type=int, default=5000, help="Gold positives target (default: 5000)")
    parser.add_argument("--gold-neg", type=int, default=5000, help="Gold negatives target (default: 5000)")
    parser.add_argument("--min-per-class", type=int, default=900, help="Minimum per symptom (default: 900)")
    parser.add_argument("--max-per-class", type=int, default=1000, help="Maximum per symptom (default: 1000)")
    parser.add_argument("--oversample-low-count", action="store_true", help="Oversample rare symptoms in data/auto_labeled.jsonl and update label_counts.csv")
    parser.add_argument("--downsample-high-count", action="store_true", help="Downsample frequent symptoms in data/auto_labeled.jsonl to <= max per class and update label_counts.csv")
    parser.add_argument("--dedupe-symptoms", action="store_true", help="Deduplicate symptoms.csv against itself, update synonyms.json, and write unique symptoms to symptoms.csv and symptoms.json")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data dir not found: {args.data_dir}")
    for name in ["symptoms.json", "synonyms.json"]:
        if not os.path.exists(os.path.join(args.data_dir, name)):
            raise FileNotFoundError(f"Missing {name} in {args.data_dir}")

    if args.oversample_low_count:
        print("Oversampling low-count symptoms in existing silver set...")
        auto_path, counts_path = oversample_low_count_sentences(
            data_dir=args.data_dir,
            auto_labeled_filename="auto_labeled.jsonl",
            counts_filename="label_counts.csv",
            min_per_class=args.min_per_class,
            max_per_class=args.max_per_class,
        )
        print(f"Updated: {auto_path}")
        print(f"Counts:  {counts_path}")
    elif args.downsample_high_count:
        print("Downsampling high-count symptoms in existing silver set...")
        auto_path, counts_path = downsample_high_count_sentences(
            data_dir=args.data_dir,
            auto_labeled_filename="auto_labeled.jsonl",
            counts_filename="label_counts.csv",
            max_per_class=args.max_per_class,
        )
        print(f"Updated: {auto_path}")
        print(f"Counts:  {counts_path}")
    elif args.dedupe_symptoms:
        print("Deduplicating symptoms and updating synonyms...")
        csv_path, json_path, syn_path = deduplicate_symptoms_and_update_synonyms(
            data_dir=args.data_dir,
            csv_filename="symptoms.csv",
            json_filename="symptoms.json",
            synonyms_filename="synonyms.json",
            similarity_threshold=90,
        )
        print(f"Symptoms CSV:  {csv_path}")
        print(f"Symptoms JSON: {json_path}")
        print(f"Synonyms JSON: {syn_path}")
    elif args.make_datasets:
        print("Generating silver and gold datasets...")
        silver_path, counts_path, gold_path = generate_silver_and_gold(
            args.input,
            data_dir=args.data_dir,
            silver_pos_target=args.silver_pos,
            silver_neg_target=args.silver_neg,
            gold_pos_target=args.gold_pos,
            gold_neg_target=args.gold_neg,
            max_per_class=args.max_per_class,
            min_per_class=args.min_per_class,
            fuzzy_cutoff=args.fuzzy_cutoff,
            fuzzy_max_ngram=args.fuzzy_max_ngram,
        )
        print(f"Silver: {silver_path}")
        print(f"Counts: {counts_path}")
        print(f"Gold:   {gold_path}")
    else:
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
