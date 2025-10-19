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
BIO_JSONL  = BIO_ROOT / "train.jsonl"                # tokens + tags [web:23]

for p in (SILVER_POS, SILVER_NEG, BIO_JSONL):
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}\nCWD={Path.cwd()}")  # fast fail [web:54]

tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1", use_fast=True)  # uses fast offsets [web:54]


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
    from collections import defaultdict, deque
    spans_map = defaultdict(deque)
    for ex in pos_data:
        spans_map[ex["text"]].append(normalize_spans(ex.get("spans", [])))

    rng = random.Random(seed)
    candidates = [ex for ex in bio_data if ex["text"] in spans_map and len(spans_map[ex['text']]) > 0]
    rng.shuffle(candidates)
    shown = 0

    for ex in candidates:
        text = ex["text"]
        tokens = ex["tokens"]
        tags = ex["tags"]
        gold_spans = spans_map[text][0]  # use the next available set for this text
        offsets, re_tokens = token_offsets(text)

        assert tokens == re_tokens, "Token list mismatch; ensure same tokenizer and settings were used."

        inside = inside_any_span(offsets, gold_spans)

        # Assert B/I exactly on inside indices, with B at first index of each span
        # Build per-span index lists
        per_span_idxs = []
        for (s, e, _) in gold_spans:
            idxs = [i for i,(ts,te) in enumerate(offsets) if ts >= s and te <= e and te > ts]
            if idxs:
                per_span_idxs.append(sorted(idxs))

        # Assertions
        # 1) No B/I outside any span
        outside_idxs = [i for i in range(len(tokens)) if i not in inside]
        for i in outside_idxs:
            assert tags[i] == "O", f"Non-span token tagged {tags[i]} at idx {i}: {tokens[i]}"

        # 2) Each span has B then I...I
        for idxs in per_span_idxs:
            if not idxs:
                continue
            assert tags[idxs[0]] == "B-SYM", f"Expected B-SYM at start of span but got {tags[idxs[0]]} ({tokens[idxs[0]]})"
            for j in idxs[1:]:
                assert tags[j] == "I-SYM", f"Expected I-SYM inside span but got {tags[j]} ({tokens[j]})"

        # Pretty print sample
        if shown < k:
            ann = []
            for i, t in enumerate(tokens):
                marker = "[IN]" if i in inside else "   "
                ann.append(f"{t}|{tags[i]}{marker}")
            print(f"\nTEXT: {text}\nTOKENS:\n" + " ".join(ann))
            shown += 1

        # consume this span set so duplicates use the next one
        spans_map[text].popleft()
        if shown >= k:
            break

    print(f"Checked {shown} positive examples with strict alignment assertions.")

def check_negatives(bio_data, neg_data, k=5, seed=7):
    neg_texts = set(ex["text"] for ex in neg_data)
    neg_bio = [ex for ex in bio_data if ex["text"] in neg_texts]
    rng = random.Random(seed)
    rng.shuffle(neg_bio)

    shown = 0
    for ex in neg_bio:
        tokens = ex["tokens"]
        tags = ex["tags"]
        assert all(t == "O" for t in tags), "Found non-O tag in a negative example."
        if shown < k:
            print("\nNEGATIVE SAMPLE:")
            print("TOKENS:", " ".join(tokens))
            print("TAGS:  ", " ".join(tags))
            shown += 1
        if shown >= k:
            break
    print(f"Checked {shown} negative examples with all-O assertions.")


if __name__ == "__main__":
    check_positives(bio, pos, k=5, seed=42)
    check_negatives(bio, neg, k=5, seed=42)
