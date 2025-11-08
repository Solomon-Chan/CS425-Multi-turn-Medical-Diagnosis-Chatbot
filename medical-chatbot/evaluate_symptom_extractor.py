"""Evaluate the fine-tuned SentenceTransformer symptom extractor.

This script replaces the previous token-classification evaluation by
leveraging the local SentenceTransformer model trained for symptom
matching. It reports macro, micro, entity, and token-level precision,
recall, F1, and a binary cross-entropy loss against the canonical symptom
vocabulary.

Input data must be a JSONL file where each record contains:

    {
        "text": "patient utterance",
        "spans": [
            [start_char, end_char, "canonical symptom", score?],
            ...
        ]
    }

The canonical symptom strings should align with the entries used to train
the SentenceTransformer model (e.g. the rows in `data/symptoms.csv`).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sentence_transformers import util

# Configure import path so we can reuse the symptom extractor implementation.
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

import sys

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from symptom_extractor import MLSymptomExtractor  # type: ignore


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def normalize_label(label: str) -> str:
    """Lower-case and collapse whitespace for consistent label comparison."""

    return re.sub(r"\s+", " ", label.strip().lower())


def load_jsonl(path: Path, max_examples: Optional[int] = None) -> List[dict]:
    """Load records from a JSONL file, optionally truncating to a max size."""

    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)
            if max_examples is not None and len(records) >= max_examples:
                break
    return records


def tokenise_with_offsets(text: str) -> List[Dict[str, int]]:
    """Simple regex-based tokeniser that returns per-token spans."""

    tokens: List[Dict[str, int]] = []
    for match in TOKEN_PATTERN.finditer(text):
        tokens.append({
            "token": match.group(),
            "start": match.start(),
            "end": match.end(),
        })
    return tokens


def coerce_spans(sample: dict, text: str) -> List[Dict[str, object]]:
    """Normalise span annotations in a sample."""

    spans_raw = sample.get("spans")
    if spans_raw is None:
        raise ValueError(
            "Evaluation record missing 'spans'. Supply character-level symptom spans "
            "aligned with canonical symptom names."
        )

    spans: List[Dict[str, object]] = []
    for item in spans_raw:
        if isinstance(item, dict):
            start = int(item.get("start", 0))
            end = int(item.get("end", 0))
            label = item.get("label") or item.get("symptom") or item.get("text") or ""
        else:
            start = int(item[0])
            end = int(item[1])
            label = item[2] if len(item) > 2 else ""
        label_str = str(label).strip()
        if not label_str:
            label_str = text[start:end]
        spans.append({"start": start, "end": end, "label": label_str})
    return spans


def spans_to_bio(
    tokens: Sequence[Dict[str, int]],
    spans: Sequence[Dict[str, object]],
) -> Tuple[List[str], List[Tuple[int, int, str]]]:
    """Convert spans into BIO tags and token-indexed entity tuples."""

    tags = ["O"] * len(tokens)
    entities: List[Tuple[int, int, str]] = []

    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        label = normalize_label(str(span.get("label", "sym"))) or "sym"

        covered: List[int] = []
        for idx, tok in enumerate(tokens):
            if tok["end"] <= start or tok["start"] >= end:
                continue
            covered.append(idx)

        if not covered:
            continue

        tags[covered[0]] = "B-SYM"
        for idx in covered[1:]:
            tags[idx] = "I-SYM"
        entities.append((covered[0], covered[-1], label))

    return tags, entities


def build_predicted_spans(text: str, labels: Iterable[str]) -> List[Dict[str, object]]:
    """Locate predicted canonical strings inside the original text."""

    text_norm = text.lower()
    spans: List[Dict[str, object]] = []
    for label in labels:
        if not label:
            continue
        label_norm = normalize_label(label)
        if not label_norm:
            continue
        pattern = re.escape(label_norm)
        for match in re.finditer(pattern, text_norm):
            spans.append({
                "start": match.start(),
                "end": match.end(),
                "label": label,
            })
    return spans


def entity_set(entities: Sequence[Tuple[int, int, str]]) -> Set[Tuple[int, int, str]]:
    return {(start, end, label) for start, end, label in entities}


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SentenceTransformer-based symptom extraction",
    )

    default_model = PROJECT_ROOT / "models" / "medical_symptom_matcher"
    default_data = PROJECT_ROOT / "data" / "auto_labeled.jsonl"
    default_symptoms = PROJECT_ROOT / "data" / "symptoms.csv"

    parser.add_argument("--model-dir", type=Path, default=default_model,
                        help="Path to the fine-tuned SentenceTransformer model directory")
    parser.add_argument("--symptoms-path", type=Path, default=default_symptoms,
                        help="CSV file containing canonical symptom strings")
    parser.add_argument("--valid-path", type=Path, default=default_data,
                        help="JSONL evaluation file with span annotations")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Optional limit on the number of samples to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Similarity threshold for accepting predictions")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Maximum number of symptoms returned per sample")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n=== Symptom Extractor Evaluation ===")
    print(f"Model directory    : {args.model_dir}")
    print(f"Symptoms catalog   : {args.symptoms_path}")
    print(f"Evaluation dataset : {args.valid_path}")
    if args.max_examples is not None:
        print(f"Max examples       : {args.max_examples}")
    print(f"Threshold / Top-K  : {args.threshold} / {args.top_k}\n")

    if not args.valid_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {args.valid_path}")
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    records = load_jsonl(args.valid_path, args.max_examples)
    if not records:
        raise ValueError("Evaluation dataset is empty.")

    extractor = MLSymptomExtractor(
        model_path=str(args.model_dir),
        symptoms_path=str(args.symptoms_path),
        threshold=args.threshold,
        top_k=args.top_k,
    )

    canonical_symptoms = extractor.canonical_symptoms
    label_to_index = {normalize_label(symptom): idx for idx, symptom in enumerate(canonical_symptoms)}

    print(f"Loaded {len(canonical_symptoms)} canonical symptoms")
    print(f"Evaluating {len(records)} samples\n")

    micro_tp = micro_fp = micro_fn = 0
    entity_tp = entity_fp = entity_fn = 0
    macro_precisions: List[float] = []
    macro_recalls: List[float] = []
    macro_f1s: List[float] = []
    losses: List[float] = []

    token_true: List[List[str]] = []
    token_pred: List[List[str]] = []
    debug_samples: List[Tuple[str, List[str], List[str]]] = []

    for record in records:
        text = record.get("text", "").strip()
        if not text:
            continue

        tokens = tokenise_with_offsets(text)
        if not tokens:
            continue

        gold_spans = coerce_spans(record, text)
        gold_tags, gold_entities = spans_to_bio(tokens, gold_spans)
        gold_labels = {normalize_label(span["label"]) for span in gold_spans if span.get("label")}

        extracted = extractor.extract(text, threshold=args.threshold, top_k=args.top_k)
        predicted_labels = [item.symptom for item in extracted]
        pred_label_set = {normalize_label(sym) for sym in predicted_labels if sym}

        pred_spans = build_predicted_spans(text, predicted_labels)
        pred_tags, pred_entities = spans_to_bio(tokens, pred_spans)

        token_true.append(gold_tags)
        token_pred.append(pred_tags)

        intersection = gold_labels & pred_label_set
        micro_tp += len(intersection)
        micro_fp += max(0, len(pred_label_set) - len(intersection))
        micro_fn += max(0, len(gold_labels) - len(intersection))

        precision_sample = safe_div(len(intersection), len(pred_label_set))
        recall_sample = safe_div(len(intersection), len(gold_labels))
        f1_sample = safe_div(2 * precision_sample * recall_sample, precision_sample + recall_sample)

        macro_precisions.append(precision_sample)
        macro_recalls.append(recall_sample)
        macro_f1s.append(f1_sample)

        gold_entity_set = entity_set(gold_entities)
        pred_entity_set = entity_set(pred_entities)
        entity_tp += len(gold_entity_set & pred_entity_set)
        entity_fp += len(pred_entity_set - gold_entity_set)
        entity_fn += len(gold_entity_set - pred_entity_set)

        if len(debug_samples) < 3 and gold_labels != pred_label_set:
            debug_samples.append((text, sorted(gold_labels), sorted(pred_label_set)))

        with torch.no_grad():
            sentence_embedding = extractor.model.encode(text, convert_to_tensor=True)
            similarities = util.cos_sim(sentence_embedding, extractor.symptom_embeddings)[0]
        probabilities = torch.sigmoid(similarities)
        target = torch.zeros_like(probabilities)
        for label in gold_labels:
            idx = label_to_index.get(label)
            if idx is not None:
                target[idx] = 1.0
        loss = F.binary_cross_entropy(probabilities, target).item()
        losses.append(loss)

    micro_precision = safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    macro_precision = statistics.mean(macro_precisions) if macro_precisions else 0.0
    macro_recall = statistics.mean(macro_recalls) if macro_recalls else 0.0
    macro_f1 = statistics.mean(macro_f1s) if macro_f1s else 0.0

    entity_precision = safe_div(entity_tp, entity_tp + entity_fp)
    entity_recall = safe_div(entity_tp, entity_tp + entity_fn)
    entity_f1 = safe_div(2 * entity_precision * entity_recall, entity_precision + entity_recall)

    token_precision = precision_score(token_true, token_pred, zero_division=0)
    token_recall = recall_score(token_true, token_pred, zero_division=0)
    token_f1 = f1_score(token_true, token_pred, zero_division=0)
    token_report = classification_report(token_true, token_pred, digits=4, zero_division="warn")

    average_loss = statistics.mean(losses) if losses else 0.0

    print("=== Aggregate Metrics ===")
    print(f"Micro Precision : {micro_precision:.4f}")
    print(f"Micro Recall    : {micro_recall:.4f}")
    print(f"Micro F1        : {micro_f1:.4f}\n")

    print(f"Macro Precision : {macro_precision:.4f}")
    print(f"Macro Recall    : {macro_recall:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}\n")

    print(f"Entity Precision: {entity_precision:.4f}")
    print(f"Entity Recall   : {entity_recall:.4f}")
    print(f"Entity F1       : {entity_f1:.4f}\n")

    print(f"Token Precision : {token_precision:.4f}")
    print(f"Token Recall    : {token_recall:.4f}")
    print(f"Token F1        : {token_f1:.4f}\n")

    print(f"Binary CE Loss  : {average_loss:.6f}\n")

    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "ner_eval.txt"

    with report_path.open("w", encoding="utf-8") as out:
        out.write("Symptom Extractor Evaluation\n")
        out.write("============================\n\n")

        out.write("Micro-level metrics:\n")
        out.write(f"  Precision: {micro_precision:.4f}\n")
        out.write(f"  Recall   : {micro_recall:.4f}\n")
        out.write(f"  F1       : {micro_f1:.4f}\n\n")

        out.write("Macro-level metrics:\n")
        out.write(f"  Precision: {macro_precision:.4f}\n")
        out.write(f"  Recall   : {macro_recall:.4f}\n")
        out.write(f"  F1       : {macro_f1:.4f}\n\n")

        out.write("Entity-level metrics:\n")
        out.write(f"  Precision: {entity_precision:.4f}\n")
        out.write(f"  Recall   : {entity_recall:.4f}\n")
        out.write(f"  F1       : {entity_f1:.4f}\n\n")

        out.write("Token-level metrics (seqeval):\n")
        out.write(str(token_report))
        out.write("\n")
        out.write(f"  Precision: {token_precision:.4f}\n")
        out.write(f"  Recall   : {token_recall:.4f}\n")
        out.write(f"  F1       : {token_f1:.4f}\n\n")

        out.write(f"Binary cross-entropy loss: {average_loss:.6f}\n")

        if debug_samples:
            out.write("\nExamples with mismatched predictions:\n")
            for text, gold, pred in debug_samples:
                out.write("----------------------------------------\n")
                out.write(f"Text: {text}\n")
                out.write(f"Gold: {gold}\n")
                out.write(f"Pred: {pred}\n")

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()

