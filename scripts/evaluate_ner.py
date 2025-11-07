# scripts/evaluate_ner.py

import json
import torch
import sys
import argparse
from pathlib import Path
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import config

LABEL_MAP = config.LABEL_MAP
ID2LABEL = config.ID2LABEL

def predict_tags(tokens, tokenizer, model):
    """Predict BIO tags by tokenizing with is_split_into_words=True so
    labels align exactly to the provided `tokens` list.

    This avoids reconstructing text and re-tokenizing differently than the
    gold tokenization, which previously caused off-by-one and boundary
    mismatches (no entity overlap).
    """
    # Tokenize the provided tokens as words so tokenizer.word_ids() maps pieces
    encoded = tokenizer(tokens,
                        is_split_into_words=True,
                        return_tensors="pt",
                        truncation=True,
                        max_length=config.MAX_LENGTH,
                        return_special_tokens_mask=True)

    # Forward pass (only pass expected args)
    with torch.no_grad():
        outputs = model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
    pred_ids = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    if isinstance(pred_ids[0], list):
        pred_ids = pred_ids[0]

    # Map model token predictions back to original word tokens using word_ids
    word_ids = encoded.word_ids(batch_index=0)
    pred_labels = []
    previous_word_idx = None
    for token_index, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            label_id = pred_ids[token_index]
            label = config.LABEL_MAP.get(label_id, "O")
            # Fix invalid I- tags that appear without B- before
            if label == "I-SYM" and (not pred_labels or pred_labels[-1] == "O"):
                label = "B-SYM"
            pred_labels.append(label)
            previous_word_idx = word_idx

    # Safety: ensure length matches original token list
    if len(pred_labels) < len(tokens):
        pred_labels.extend(["O"] * (len(tokens) - len(pred_labels)))
    elif len(pred_labels) > len(tokens):
        pred_labels = pred_labels[:len(tokens)]

    return pred_labels

def main():
    parser = argparse.ArgumentParser(description="Evaluate NER model with tokenizer-aligned gold tags")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples processed (for faster testing)")
    parser.add_argument("--model-dir", type=str, default=str((Path(__file__).resolve().parent.parent / "models" / "biobert_ner")), help="Path to model directory")
    parser.add_argument("--valid-path", type=str, default=str((Path(__file__).resolve().parent.parent / "data" / "valid.jsonl")), help="Path to validation jsonl")
    args = parser.parse_args()

    print("Starting NER evaluation...")
    
    # Resolve model and validation paths
    model_dir = Path(args.model_dir)
    print(f"Using model from: {model_dir}")
    valid_path = Path(args.valid_path)
    if not valid_path.exists():
        # Fallbacks
        for p in [config.DATA_DIR / "valid.jsonl", config.DATA_DIR / "validation.jsonl", config.BIO_OUTPUTS_DIR / "validation.jsonl"]:
            if p.exists():
                valid_path = p
                break
    print(f"Using validation data from: {valid_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Validation file not found at {valid_path}")

    report_path = Path("reports") / "ner_eval.txt"
    
    # For storing all tokens for analysis
    all_tokens = []
    report_path = Path("reports") / "ner_eval.txt"
    
    if not valid_path.exists():
        # Try bio_outputs as fallback
        valid_path = config.BIO_OUTPUTS_DIR / "validation.jsonl"
        if not valid_path.exists():
            raise FileNotFoundError(f"Validation file not found. Checked {config.DATA_DIR / 'validation.jsonl'} and {config.BIO_OUTPUTS_DIR / 'validation.jsonl'}")
    
    # Always generate fresh predictions
    print(f"Generating predictions using model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.eval()

    true_tags = []
    pred_tags = []
    # For loss aggregation
    token_losses = []  # per-token losses
    sentence_losses = []  # average loss per sentence (macro proxy)
    entity_token_losses = []  # loss restricted to entity tokens
    ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
    
    # Generate predictions on-the-fly; align gold to the SAME tokenizer encoding
    with open(str(valid_path), "r", encoding="utf-8") as f:
        processed = 0
        for line in f:
            if not line.strip():
                continue
            if args.max_examples is not None and processed >= args.max_examples:
                break
            x = json.loads(line)
            tokens = x["tokens"]
            gold = x["tags"] if "tags" in x else x.get("labels", ["O"] * len(tokens))

            # Encode the token list as words so tokenizer.word_ids() maps to indices
            encoded = tokenizer(tokens,
                                is_split_into_words=True,
                                return_tensors="pt",
                                truncation=True,
                                max_length=config.MAX_LENGTH,
                                return_special_tokens_mask=True)

            # Build gold labels aligned to the encoded word pieces
            word_ids = encoded.word_ids(batch_index=0)
            gold_aligned = []
            gold_ids_for_loss = []  # aligned to input_ids length with -100 for special/subword positions
            prev_word_idx = None
            for token_index, word_idx in enumerate(word_ids):
                if word_idx is None:
                    gold_ids_for_loss.append(-100)
                    continue
                if word_idx != prev_word_idx:
                    # Use the gold label for that original token index
                    lbl = gold[word_idx] if word_idx < len(gold) else "O"
                    gold_aligned.append(lbl)
                    # Map gold label to id for loss
                    inv_map = {v: k for k, v in config.LABEL_MAP.items()}
                    gold_id = inv_map.get(lbl, inv_map.get("O", 0))
                    gold_ids_for_loss.append(gold_id)
                    prev_word_idx = word_idx
                else:
                    # subword: ignore in loss
                    gold_ids_for_loss.append(-100)

            # Predict using the model on the exact same encoding
            with torch.no_grad():
                outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
            pred_ids = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
            if isinstance(pred_ids[0], list):
                pred_ids = pred_ids[0]

            # Map predictions back to original token indices using word_ids
            pred_aligned = []
            prev_word_idx = None
            for token_index, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                if word_idx != prev_word_idx:
                    label_id = pred_ids[token_index]
                    label = config.LABEL_MAP.get(label_id, "O")
                    if label == "I-SYM" and (not pred_aligned or pred_aligned[-1] == "O"):
                        label = "B-SYM"
                    pred_aligned.append(label)
                    prev_word_idx = word_idx

            # Ensure alignment length equals original tokens length
            if len(gold_aligned) < len(tokens):
                gold_aligned.extend(["O"] * (len(tokens) - len(gold_aligned)))
            elif len(gold_aligned) > len(tokens):
                gold_aligned = gold_aligned[:len(tokens)]

            if len(pred_aligned) < len(tokens):
                pred_aligned.extend(["O"] * (len(tokens) - len(pred_aligned)))
            elif len(pred_aligned) > len(tokens):
                pred_aligned = pred_aligned[:len(tokens)]

            true_tags.append(gold_aligned)
            pred_tags.append(pred_aligned)
            all_tokens.append(tokens)  # Store tokens for analysis
            # Compute losses
            logits = outputs.logits.squeeze(0)  # seq_len x num_labels
            gold_tensor = torch.tensor(gold_ids_for_loss, dtype=torch.long)
            # Align shapes if needed
            if logits.size(0) != len(gold_ids_for_loss):
                # pad/truncate gold to logits length
                if len(gold_ids_for_loss) < logits.size(0):
                    gold_tensor = torch.cat([gold_tensor, torch.full((logits.size(0)-len(gold_ids_for_loss),), -100, dtype=torch.long)])
                else:
                    gold_tensor = gold_tensor[:logits.size(0)]
            per_token_loss = ce_loss(logits, gold_tensor)  # length seq_len
            valid_mask = (gold_tensor != -100)
            if valid_mask.any():
                # token-level aggregation
                token_losses.extend(per_token_loss[valid_mask].tolist())
                # sentence-level average
                sentence_losses.append(per_token_loss[valid_mask].mean().item())
                # entity-token-only loss
                entity_mask = torch.tensor([1 if (t != -100 and (g != 0 and config.LABEL_MAP.get(g, 'O') != 'O')) else 0 for t, g in zip(gold_tensor.tolist(), gold_tensor.tolist())], dtype=torch.bool)
                if entity_mask.any():
                    entity_token_losses.extend(per_token_loss[entity_mask].tolist())
            processed += 1
    # Print detailed debug info
    print(f"\nProcessed {len(true_tags)} sentences")
    print("\nDetailed Sample Analysis:")
    for i, (tokens, gold, pred) in enumerate(zip(all_tokens[:3], true_tags[:3], pred_tags[:3])):
        print(f"\nExample {i+1}:")
        print("Text:", " ".join(tokens))
        print("\nToken-by-Token Comparison:")
        for t, g, p in zip(tokens, gold, pred):
            if g != p:
                print(f"Token: {t:15} Gold: {g:10} Pred: {p:10} {'[MISMATCH]' if g != p else ''}")
        print("\nFull sequences:")
        print("Gold:", gold)
        print("Pred:", pred)
        print("-" * 80)

    # Analyze tag distributions
    gold_tag_dist = {"B-SYM": 0, "I-SYM": 0, "O": 0}
    pred_tag_dist = {"B-SYM": 0, "I-SYM": 0, "O": 0}
    
    for tags in true_tags:
        for tag in tags:
            gold_tag_dist[tag] = gold_tag_dist.get(tag, 0) + 1
    
    for tags in pred_tags:
        for tag in tags:
            pred_tag_dist[tag] = pred_tag_dist.get(tag, 0) + 1

    print("\nTag Distribution Analysis:")
    print("Gold Tags:", gold_tag_dist)
    print("Pred Tags:", pred_tag_dist)
    
    # Count B-SYM tags and analyze patterns
    gold_b_count = sum(1 for tags in true_tags for tag in tags if tag == "B-SYM")
    pred_b_count = sum(1 for tags in pred_tags for tag in tags if tag == "B-SYM")
    
    # Analyze B-I-O patterns
    print("\nB-I-O Pattern Analysis:")
    invalid_gold = sum(1 for tags in true_tags for i, tag in enumerate(tags) 
                      if tag == "I-SYM" and (i == 0 or tags[i-1] not in ["B-SYM", "I-SYM"]))
    invalid_pred = sum(1 for tags in pred_tags for i, tag in enumerate(tags) 
                      if tag == "I-SYM" and (i == 0 or tags[i-1] not in ["B-SYM", "I-SYM"]))
    print(f"Number of B-SYM tags in gold: {gold_b_count}")
    print(f"Number of B-SYM tags in predictions: {pred_b_count}")

    # Convert BIO tag sequences to entity spans for entity-level diagnostics
    def bio_to_entities(tags):
        """Convert a list of BIO tags (e.g. B-SYM, I-SYM, O) to a list of entities.
        Each entity is a tuple: (start_index, end_index_inclusive, label_type).
        """
        ents = []
        i = 0
        while i < len(tags):
            tag = tags[i]
            if tag.startswith("B-"):
                label = tag.split("-", 1)[1]
                start = i
                j = i + 1
                while j < len(tags) and tags[j] == f"I-{label}":
                    j += 1
                end = j - 1
                ents.append((start, end, label))
                i = j
            else:
                i += 1
        return ents

    total_gold_ents = 0
    total_pred_ents = 0
    tp = 0
    fp = 0
    fn = 0
    sample_fp_examples = []
    sample_fn_examples = []

    for tokens, gold, pred in zip(all_tokens, true_tags, pred_tags):
        gents = set(bio_to_entities(gold))
        pents = set(bio_to_entities(pred))
        total_gold_ents += len(gents)
        total_pred_ents += len(pents)
        tp += len(gents & pents)
        fp += len(pents - gents)
        fn += len(gents - pents)
        # Collect some example mismatches for inspection
        if len(sample_fp_examples) < 3 and len(pents - gents) > 0:
            sample_fp_examples.append((tokens, gold, pred, list(pents - gents)))
        if len(sample_fn_examples) < 3 and len(gents - pents) > 0:
            sample_fn_examples.append((tokens, gold, pred, list(gents - pents)))

    print("\nEntity-level diagnostics:")
    print(f"Total gold entities : {total_gold_ents}")
    print(f"Total pred entities : {total_pred_ents}")
    print(f"True positives      : {tp}")
    print(f"False positives     : {fp}")
    print(f"False negatives     : {fn}")
    if total_pred_ents > 0:
        ent_precision = tp / total_pred_ents
    else:
        ent_precision = 0.0
    if total_gold_ents > 0:
        ent_recall = tp / total_gold_ents
    else:
        ent_recall = 0.0
    if ent_precision + ent_recall > 0:
        ent_f1 = 2 * ent_precision * ent_recall / (ent_precision + ent_recall)
    else:
        ent_f1 = 0.0
    print(f"Entity Precision: {ent_precision:.4f}")
    print(f"Entity Recall   : {ent_recall:.4f}")
    print(f"Entity F1       : {ent_f1:.4f}")

    # Show a few FP/FN examples
    if sample_fp_examples:
        print("\nSample false positives (predicted entity not in gold):")
        for tokens, gold, pred, pents in sample_fp_examples:
            print("Text:", " ".join(tokens))
            print("Predicted extra entities:", pents)
            print("Gold sequence:", gold)
            print("Pred sequence:", pred)
            print("-" * 60)
    if sample_fn_examples:
        print("\nSample false negatives (gold entity missed by model):")
        for tokens, gold, pred, gents in sample_fn_examples:
            print("Text:", " ".join(tokens))
            print("Missed gold entities:", gents)
            print("Gold sequence:", gold)
            print("Pred sequence:", pred)
            print("-" * 60)

    # Compute metrics, keep classification_report for token-level view
    report = classification_report(true_tags, pred_tags, digits=4, zero_division="warn")
    print("\nEvaluation Report:")
    print(report)
    
    # Token-level (micro) metrics
    precision = precision_score(true_tags, pred_tags, zero_division="warn")
    recall = recall_score(true_tags, pred_tags, zero_division="warn")
    f1 = f1_score(true_tags, pred_tags, zero_division="warn")
    # Token-level macro metrics
    precision_macro = precision_score(true_tags, pred_tags, average='macro', zero_division="warn")
    recall_macro = recall_score(true_tags, pred_tags, average='macro', zero_division="warn")
    f1_macro = f1_score(true_tags, pred_tags, average='macro', zero_division="warn")
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    # Aggregate losses
    token_loss_avg = sum(token_losses)/len(token_losses) if token_losses else 0.0
    sentence_loss_avg = sum(sentence_losses)/len(sentence_losses) if sentence_losses else 0.0
    entity_token_loss_avg = sum(entity_token_losses)/len(entity_token_losses) if entity_token_losses else 0.0

    # Save report text
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(report_path), "w") as out:
        out.write("NER Evaluation (BioBERT)\n")
        out.write("========================\n\n")
        out.write("Token-level metrics (seqeval):\n")
        out.write(str(report))
        out.write("\n")
        out.write(f"Token Precision (micro): {precision:.4f}\n")
        out.write(f"Token Recall    (micro): {recall:.4f}\n")
        out.write(f"Token F1        (micro): {f1:.4f}\n")
        out.write(f"Token Precision (macro): {precision_macro:.4f}\n")
        out.write(f"Token Recall    (macro): {recall_macro:.4f}\n")
        out.write(f"Token F1        (macro): {f1_macro:.4f}\n")
        out.write(f"Token Loss (avg per token): {token_loss_avg:.6f}\n\n")

        out.write("Entity-level metrics:\n")
        out.write(f"Entity Precision: {ent_precision:.4f}\n")
        out.write(f"Entity Recall   : {ent_recall:.4f}\n")
        out.write(f"Entity F1       : {ent_f1:.4f}\n")
        out.write(f"Entity-token Loss (avg per entity token): {entity_token_loss_avg:.6f}\n\n")

        out.write("Macro-level metrics:\n")
        out.write(f"Macro (sentence-avg) Loss: {sentence_loss_avg:.6f}\n")
        out.write(f"Macro Precision (token-level macro): {precision_macro:.4f}\n")
        out.write(f"Macro Recall    (token-level macro): {recall_macro:.4f}\n")
        out.write(f"Macro F1        (token-level macro): {f1_macro:.4f}\n")

if __name__ == "__main__":
    main()
