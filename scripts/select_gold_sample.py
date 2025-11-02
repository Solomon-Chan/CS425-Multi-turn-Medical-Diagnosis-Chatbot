# scripts/select_gold_sample.py
import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Set

NEG_TRIGGERS = {"no","not","denies","without","never","free of","absence of","negative for","rule out","r/o","might","could","possible","possibly","likely"}

def load_jsonl(path: Path) -> List[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def build_text_to_labels(silver_pos: List[dict]) -> Dict[str, Set[str]]:
    # Map sentence text -> set of canonical symptom labels from auto_labeled_sample.jsonl
    t2labels: Dict[str, Set[str]] = defaultdict(set)
    for ex in silver_pos:
        text = ex["text"]
        spans = ex.get("spans", [])
        for sp in spans:
            if len(sp) >= 3:
                label = str(sp[2]).strip().lower()
                t2labels[text].add(label)
    return t2labels

def b_count(tags: List[str]) -> int:
    return sum(1 for t in tags if t == "B-SYM")

def has_negation(text: str) -> bool:
    lt = text.lower()
    return any(trig in lt for trig in NEG_TRIGGERS)

def select_stratified(
    bio: List[dict],
    t2labels: Dict[str, Set[str]],
    n: int,
    seed: int = 42,
    neg_frac: float = 0.15,
    ambig_frac: float = 0.30,
    rare_frac: float = 0.35,
):
    rng = random.Random(seed)

    # Compute label frequencies from silver to identify rare labels
    label_counter = Counter()
    for labels in t2labels.values():
        label_counter.update(labels)
    labels_sorted = sorted(label_counter.items(), key=lambda x: x[1])
    # Rare = bottom quartile by frequency (at least 1 occurrence)
    if labels_sorted:
        counts = [c for _, c in labels_sorted]
        q25 = counts[max(0, int(0.25 * (len(counts)-1)))]
        rare_labels = {lab for lab, c in labels_sorted if c <= max(1, q25)}
    else:
        rare_labels = set()

    # Index examples into buckets
    neg_idx, amb_idx, common_idx = [], [], []
    label_to_idxs: Dict[str, List[int]] = defaultdict(list)

    for i, ex in enumerate(bio):
        text = ex["text"]
        tags = ex["tags"]
        bsym = b_count(tags)
        labels = t2labels.get(text, set())
        # Link each canonical label to this entry for stratified draws
        for lab in labels:
            label_to_idxs[lab].append(i)

        if bsym == 0:
            neg_idx.append(i)
        else:
            ambiguous = (bsym >= 2) or has_negation(text)
            if ambiguous:
                amb_idx.append(i)
            else:
                common_idx.append(i)

    rng.shuffle(neg_idx)
    rng.shuffle(amb_idx)
    rng.shuffle(common_idx)
    for lst in label_to_idxs.values():
        rng.shuffle(lst)

    target_neg = min(len(neg_idx), round(n * neg_frac))
    target_amb = min(len(amb_idx), round(n * ambig_frac))

    # Rare label coverage: try to take one example per rare label first
    target_rare = round(n * rare_frac)
    chosen: List[int] = []
    chosen_set: Set[int] = set()

    def add_index(idx: int):
        if idx not in chosen_set:
            chosen.append(idx)
            chosen_set.add(idx)

    # 1) Rare label coverage
    rare_taken = 0
    for lab in sorted(rare_labels, key=lambda x: label_counter.get(x, 0)):
        pool = label_to_idxs.get(lab, [])
        for idx in pool:
            add_index(idx)
            rare_taken += 1
            break
        if len(chosen) >= n or rare_taken >= target_rare:
            break

    # 2) Ambiguous slice
    amb_taken = 0
    for idx in amb_idx:
        if len(chosen) >= n or amb_taken >= target_amb:
            break
        if idx not in chosen_set:
            add_index(idx)
            amb_taken += 1

    # 3) Negative slice
    neg_taken = 0
    for idx in neg_idx:
        if len(chosen) >= n or neg_taken >= target_neg:
            break
        if idx not in chosen_set:
            add_index(idx)
            neg_taken += 1

    # 4) Fill remainder from common, then any pool
    for idx in common_idx:
        if len(chosen) >= n:
            break
        add_index(idx)
    if len(chosen) < n:
        # fallback: mix from all indices
        all_idx = list(range(len(bio)))
        rng.shuffle(all_idx)
        for idx in all_idx:
            if len(chosen) >= n:
                break
            add_index(idx)

    return chosen[:n], {
        "rare_labels": sorted(list(rare_labels)),
        "label_counts": dict(label_counter),
        "taken_counts": {
            "rare": rare_taken,
            "ambiguous": amb_taken,
            "negative": neg_taken,
            "total": len(chosen[:n]),
        },
    }

def main(n: int = 150, seed: int = 42):
    # Resolve repo structure: <repo_root>/bio_outputs and <repo_root>/data
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent
    BIO_ROOT = REPO_ROOT / "bio_outputs"
    DATA_DIR = REPO_ROOT / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    bio_file = BIO_ROOT / "train.jsonl"
    # Auto-labeling writes to data/, so read from there
    silver_pos_file = DATA_DIR / "auto_labeled_sample.jsonl"
    if not silver_pos_file.exists():
        # Fallback to bio_outputs if not in data/
        silver_pos_file = BIO_ROOT / "auto_labeled_sample.jsonl"
    out_file = DATA_DIR / "validation_sample.jsonl"

    if not bio_file.exists():
        raise FileNotFoundError(f"Missing BIO file: {bio_file}")
    if not silver_pos_file.exists():
        raise FileNotFoundError(f"Missing silver positives file: {silver_pos_file}")

    bio = load_jsonl(bio_file)
    silver_pos = load_jsonl(silver_pos_file)
    t2labels = build_text_to_labels(silver_pos)

    chosen_idx, stats = select_stratified(bio, t2labels, n=n, seed=seed)

    with out_file.open("w", encoding="utf-8") as out:
        for i in chosen_idx:
            ex = bio[i]
            # Add small metadata to aid manual review
            meta = {
                "num_entities": b_count(ex["tags"]),
                "has_negation": has_negation(ex["text"]),
                "labels_from_silver": sorted(list(t2labels.get(ex["text"], set()))),
            }
            rec = {"text": ex["text"], "tokens": ex["tokens"], "tags": ex["tags"], "meta": meta}
            json.dump(rec, out, ensure_ascii=False)
            out.write("\n")

    print(f"Saved {len(chosen_idx)} examples to {out_file}")
    print("Stats:", json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
