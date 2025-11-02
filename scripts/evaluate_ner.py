# scripts/evaluate_ner.py

import json
import torch
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForTokenClassification

LABEL_MAP = {0: "B-SYM", 1: "I-SYM", 2: "O"}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

def predict_tags(tokens, tokenizer, model):
    # Rebuild sentence from tokens for model input
    text = tokenizer.convert_tokens_to_string(tokens)
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**encodings).logits
    pred_ids = torch.argmax(logits, dim=-1).squeeze().tolist()
    # If batch_singlet, squeeze, else use [0]
    if isinstance(pred_ids[0], list): # batch of len 1
        pred_ids = pred_ids[0]
    model_tokens = tokenizer.tokenize(text)
    # Align prediction length (may be off by special tokens)
    offset = len(pred_ids) - len(model_tokens)
    pred_ids = pred_ids[offset:]
    pred_labels = [ID2LABEL.get(pid, "O") for pid in pred_ids]
    # Truncate in case subword splits/merges, else pad
    pred_labels = pred_labels[:len(tokens)] + ["O"] * (len(tokens) - len(pred_labels))
    return pred_labels

def main():
    model_dir = "./biobert-ner"  # Update as needed
    valid_path = "data/validation.jsonl"   # Gold validation file
    pred_path = "data/validation_preds.jsonl"  # Your predicted tags with same structure
    report_path = "reports/ner_eval.txt"
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

    true_tags = []
    pred_tags = []
    with open(valid_path, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            tokens = x["tokens"]
            gold = x["tags"] if "tags" in x else x["labels"]  # use whichever exists
            pred = predict_tags(tokens, tokenizer, model)
            # Truncate to match tokens
            if len(pred) > len(tokens):
                pred = pred[:len(tokens)]
            elif len(pred) < len(tokens):
                pred += ["O"] * (len(tokens) - len(pred))
            true_tags.append(gold)
            pred_tags.append(pred)
    # Compute metrics
    report = classification_report(true_tags, pred_tags, digits=4)
    print(report)
    print(f"Precision: {precision_score(true_tags, pred_tags):.4f}")
    print(f"Recall   : {recall_score(true_tags, pred_tags):.4f}")
    print(f"F1       : {f1_score(true_tags, pred_tags):.4f}")

    # Save report text
    import os
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as out:
        out.write(report)
        out.write("\n")
        out.write(f"Precision: {precision_score(true_tags, pred_tags):.4f}\n")
        out.write(f"Recall   : {recall_score(true_tags, pred_tags):.4f}\n")
        out.write(f"F1       : {f1_score(true_tags, pred_tags):.4f}\n")

if __name__ == "__main__":
    main()
