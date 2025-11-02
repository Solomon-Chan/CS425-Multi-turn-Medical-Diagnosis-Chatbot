# tests/test_bio_tags.py
from pathlib import Path
import json, random
from typing import List, Tuple
from transformers import AutoTokenizer

TEST_DIR = Path(__file__).resolve().parent           # <repo_root>/tests [web:54]
REPO_ROOT = TEST_DIR.parent                          # <repo_root> [web:54]
BIO_ROOT  = REPO_ROOT / "bio_outputs"                # <repo_root>/bio_outputs [web:38]

SILVER_POS = BIO_ROOT / "auto_labeled_sample.jsonl"  # text + spans [web:38]
SILVER_NEG = BIO_ROOT / "unknown_spans.jsonl"        # text only [web:38]
# BIO_JSONL  = BIO_ROOT / "train.jsonl"                # tokens + tags [web:23]
BIO_JSONL  = BIO_ROOT / "validation_sample.jsonl"                # tokens + tags [web:23]

for p in (SILVER_POS, SILVER_NEG, BIO_JSONL):
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}\nCWD={Path.cwd()}")  # fast fail [web:54]

tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1", use_fast=True)  # uses fast offsets [web:54]~


def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

pos = load_jsonl(SILVER_POS)
neg = load_jsonl(SILVER_NEG)
bio = load_jsonl(BIO_JSONL)
print(len(pos), len(neg), len(bio))

def normalize_spans(spans: List[List]) -> List[Tuple[int,int,str]]:
    norm = []
    for sp in spans or []:
        if len(sp) >= 3:
            s, e, lab = int(sp[0]), int(sp[1]), str(sp[2]).strip().lower()
            norm.append((s, e, lab))
    return norm

def token_offsets(text: str):
    enc = tok(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False
    )
    return enc["offset_mapping"], tok.convert_ids_to_tokens(enc["input_ids"])

def inside_any_span(offsets, spans):
    inside = set()
    for (s, e, _) in spans:
        for i, (ts, te) in enumerate(offsets):
            if ts >= s and te <= e and te > ts:
                inside.add(i)
    return inside

def check_positives(bio_data, pos_data, k=5, seed=13):
    # Build multimap from text -> list of span lists to handle duplicates
    from collections import defaultdict, deque, Counter
    spans_map = defaultdict(deque)
    for ex in pos_data:
        spans_map[ex["text"]].append(normalize_spans(ex.get("spans", [])))

    rng = random.Random(seed)
    candidates = [ex for ex in bio_data if ex["text"] in spans_map and len(spans_map[ex['text']]) > 0]
    rng.shuffle(candidates)
    shown = 0

    # Metrics tracking
    total_tokens = 0
    correct_tokens = 0
    total_spans = 0
    correct_spans = 0
    tag_errors = Counter()  # (predicted, expected) -> count
    
    # Per-class metrics: actual vs predicted
    class_actual = Counter()
    class_predicted = Counter()
    class_correct = Counter()

    for ex in candidates:
        text = ex["text"]
        tokens = ex["tokens"]
        tags = ex["tags"]
        gold_spans = spans_map[text][0]  # use the next available set for this text
        offsets, re_tokens = token_offsets(text)

        if tokens != re_tokens:
            print(f"WARNING: Token mismatch for text, skipping: {text[:50]}...")
            continue

        inside = inside_any_span(offsets, gold_spans)

        # Build per-span index lists
        per_span_idxs = []
        for (s, e, _) in gold_spans:
            idxs = [i for i,(ts,te) in enumerate(offsets) if ts >= s and te <= e and te > ts]
            if idxs:
                per_span_idxs.append(sorted(idxs))
                total_spans += 1

        # Check each token
        expected_tags = ["O"] * len(tokens)
        for idxs in per_span_idxs:
            if idxs:
                expected_tags[idxs[0]] = "B-SYM"
                for j in idxs[1:]:
                    expected_tags[j] = "I-SYM"

        # Evaluate accuracy
        for i in range(len(tokens)):
            total_tokens += 1
            actual_tag = expected_tags[i]
            pred_tag = tags[i]
            
            class_actual[actual_tag] += 1
            class_predicted[pred_tag] += 1
            
            if actual_tag == pred_tag:
                correct_tokens += 1
                class_correct[actual_tag] += 1
            else:
                tag_errors[(pred_tag, actual_tag)] += 1

        # Check span-level correctness (all tokens in span must be correct)
        for idxs in per_span_idxs:
            if idxs:
                span_correct = all(tags[i] == expected_tags[i] for i in idxs)
                if span_correct:
                    correct_spans += 1

        # Pretty print sample
        if shown < k:
            ann = []
            for i, t in enumerate(tokens):
                marker = "[IN]" if i in inside else "   "
                correct_mark = "✓" if tags[i] == expected_tags[i] else "✗"
                ann.append(f"{t}|{tags[i]}{marker}{correct_mark}")
            print(f"\nTEXT: {text}\nTOKENS:\n" + " ".join(ann))
            shown += 1

        # consume this span set so duplicates use the next one
        spans_map[text].popleft()
        if shown >= k:
            break

    # Calculate and print accuracy metrics
    accuracy = (correct_tokens / total_tokens * 100) if total_tokens > 0 else 0
    span_accuracy = (correct_spans / total_spans * 100) if total_spans > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"POSITIVE EXAMPLES ACCURACY REPORT")
    print(f"{'='*60}")
    print(f"Examples checked: {shown}")
    print(f"Total tokens: {total_tokens}")
    print(f"Correct tokens: {correct_tokens}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"\nSpan-level accuracy: {correct_spans}/{total_spans} ({span_accuracy:.2f}%)")
    
    print(f"\nPer-class accuracy:")
    for tag in sorted(set(class_actual.keys()) | set(class_predicted.keys())):
        correct = class_correct.get(tag, 0)
        actual = class_actual.get(tag, 0)
        pred = class_predicted.get(tag, 0)
        precision = (correct / pred * 100) if pred > 0 else 0
        recall = (correct / actual * 100) if actual > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        print(f"  {tag:8s}: {correct:5d}/{actual:5d} correct (Recall: {recall:5.2f}%, "
              f"Predicted: {pred:5d}, Precision: {precision:5.2f}%, F1: {f1:5.2f}%)")
    
    if tag_errors:
        print(f"\nMost common tag errors:")
        for (pred, actual), count in tag_errors.most_common(10):
            print(f"  {pred:8s} -> {actual:8s}: {count} times")

def check_negatives(bio_data, neg_data, k=5, seed=7):
    from collections import Counter
    neg_texts = set(ex["text"] for ex in neg_data)
    neg_bio = [ex for ex in bio_data if ex["text"] in neg_texts]
    rng = random.Random(seed)
    rng.shuffle(neg_bio)

    # Metrics tracking
    total_examples = 0
    correct_examples = 0  # examples with all O tags
    total_tokens = 0
    correct_tokens = 0
    tag_errors = Counter()  # predicted_tag -> count

    shown = 0
    for ex in neg_bio:
        tokens = ex["tokens"]
        tags = ex["tags"]
        total_examples += 1
        
        # Check if all tags are O
        all_o = all(t == "O" for t in tags)
        if all_o:
            correct_examples += 1
        
        # Track token-level accuracy
        for tag in tags:
            total_tokens += 1
            if tag == "O":
                correct_tokens += 1
            else:
                tag_errors[tag] += 1
        
        if shown < k:
            print("\nNEGATIVE SAMPLE:")
            print("TOKENS:", " ".join(tokens))
            print("TAGS:  ", " ".join(tags))
            if not all_o:
                print("WARNING: Found non-O tags in negative example!")
            shown += 1
        if shown >= k:
            break
    
    # Calculate and print accuracy metrics
    example_accuracy = (correct_examples / total_examples * 100) if total_examples > 0 else 0
    token_accuracy = (correct_tokens / total_tokens * 100) if total_tokens > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"NEGATIVE EXAMPLES ACCURACY REPORT")
    print(f"{'='*60}")
    print(f"Examples checked: {shown}")
    print(f"Total examples: {total_examples}")
    print(f"Examples with all-O tags: {correct_examples}")
    print(f"Example-level accuracy: {example_accuracy:.2f}%")
    print(f"\nTotal tokens: {total_tokens}")
    print(f"Correct tokens (O): {correct_tokens}")
    print(f"Token-level accuracy: {token_accuracy:.2f}%")
    
    if tag_errors:
        print(f"\nTag errors found (non-O tags in negative examples):")
        for tag, count in tag_errors.most_common():
            print(f"  {tag}: {count} tokens")
    else:
        print(f"\n✓ All negative examples correctly tagged as O")


if __name__ == "__main__":
    check_positives(bio, pos, k=5, seed=42)
    check_negatives(bio, neg, k=5, seed=42)
