# tests/test_infer_ner.py

import json
import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import config

LABEL_MAP = config.LABEL_MAP
ID2LABEL = config.ID2LABEL

def extract_symptoms(tokens, preds, label_map):
    """Extract symptom phrases from tokens and predictions."""
    symptoms = []
    current = []
    for token, label_id in zip(tokens, preds):
        label = label_map.get(label_id, "O")
        if label == "B-SYM":
            if current: symptoms.append(" ".join(current))
            current = [token]
        elif label == "I-SYM":
            if current: current.append(token)
        else:
            if current:
                symptoms.append(" ".join(current))
                current = []
    if current:  # flush at end
        symptoms.append(" ".join(current))
    return symptoms

def predict_tags(tokens, tokenizer, model):
    """
    Predict BIO tags for given tokens using the NER model.
    Uses the same logic as scripts/evaluate_ner.py for consistency.
    
    Args:
        tokens: List of tokens (as they appear in validation.jsonl)
        tokenizer: Tokenizer instance
        model: Fine-tuned NER model
        
    Returns:
        List of predicted BIO tags (B-SYM, I-SYM, O)
    """
    # Rebuild sentence from tokens for model input
    text = tokenizer.convert_tokens_to_string(tokens)
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=config.MAX_LENGTH)
    
    with torch.no_grad():
        logits = model(**encodings).logits
    pred_ids = torch.argmax(logits, dim=-1).squeeze().tolist()
    
    # Handle batch dimension if needed
    if isinstance(pred_ids[0], list):  # batch of len 1
        pred_ids = pred_ids[0]
    
    model_tokens = tokenizer.tokenize(text)
    
    # Align prediction length (may be off by special tokens)
    # The model outputs predictions for: [CLS] + tokens + [SEP]
    # We need to align with model_tokens (without CLS/SEP)
    offset = len(pred_ids) - len(model_tokens)
    if offset > 0:
        # Remove special tokens from the beginning
        pred_ids = pred_ids[offset:]
    
    # Convert to labels
    pred_labels = [ID2LABEL.get(pid, "O") for pid in pred_ids]
    
    # Align with original tokens length
    # Truncate in case subword splits/merges, else pad
    if len(pred_labels) > len(tokens):
        pred_labels = pred_labels[:len(tokens)]
    elif len(pred_labels) < len(tokens):
        pred_labels = pred_labels + ["O"] * (len(tokens) - len(pred_labels))
    
    return pred_labels

def predict_symptoms(sentence, model_dir=None):
    """Predict symptoms from a sentence using the fine-tuned NER model."""
    if model_dir is None:
        config.validate_paths()
        model_dir = config.MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=config.MAX_LENGTH)
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    tokens = tokenizer.tokenize(sentence)
    # If the model's output includes special tokens, strip them
    if len(preds) > len(tokens):
        offset = len(preds) - len(tokens) - 1  # Account for CLS token
        preds = preds[offset:-1]  # Remove CLS and SEP
    elif len(preds) > len(tokens):
        preds = preds[1:-1]  # Remove CLS and SEP
    # Align lengths
    preds = preds[:len(tokens)]
    return extract_symptoms(tokens, preds, LABEL_MAP)

def generate_validation_predictions(
    validation_path=None, 
    output_path=None, 
    model_dir=None
):
    """
    Generate predictions for validation set and save to validation_preds.jsonl.
    
    Args:
        validation_path: Path to validation.jsonl file
        output_path: Path to save validation_preds.jsonl
        model_dir: Path to model directory (uses config default if None)
    """
    config.validate_paths()
    
    if model_dir is None:
        model_dir = config.MODEL_PATH
    
    # Determine validation file path
    if validation_path is None:
        validation_path = config.DATA_DIR / "validation.jsonl"
        if not validation_path.exists():
            validation_path = config.BIO_OUTPUTS_DIR / "validation.jsonl"
            if not validation_path.exists():
                raise FileNotFoundError(
                    f"Validation file not found. Checked {config.DATA_DIR / 'validation.jsonl'} "
                    f"and {config.BIO_OUTPUTS_DIR / 'validation.jsonl'}"
                )
    
    # Determine output path
    if output_path is None:
        output_path = config.DATA_DIR / "validation_preds.jsonl"
    
    print(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    
    print(f"Reading validation set from: {validation_path}")
    print(f"Writing predictions to: {output_path}")
    
    predictions_written = 0
    with open(str(validation_path), "r", encoding="utf-8") as f_in:
        with open(str(output_path), "w", encoding="utf-8") as f_out:
            for line_num, line in enumerate(f_in, 1):
                if not line.strip():
                    continue
                
                try:
                    entry = json.loads(line)
                    tokens = entry["tokens"]
                    
                    # Predict tags for this example
                    predicted_tags = predict_tags(tokens, tokenizer, model)
                    
                    # Create output entry with same structure as input
                    output_entry = {
                        "text": entry.get("text", ""),
                        "tokens": tokens,
                        "tags": predicted_tags,  # Predicted tags
                        "gold_tags": entry.get("tags", entry.get("labels", [])),  # Keep gold tags for reference
                    }
                    
                    # Keep meta if present
                    if "meta" in entry:
                        output_entry["meta"] = entry["meta"]
                    
                    json.dump(output_entry, f_out, ensure_ascii=False)
                    f_out.write("\n")
                    predictions_written += 1
                    
                    if line_num % 10 == 0:
                        print(f"Processed {line_num} examples...", end="\r")
                
                except Exception as e:
                    print(f"\nError processing line {line_num}: {e}")
                    continue
    
    print(f"\n✓ Generated predictions for {predictions_written} examples")
    print(f"✓ Saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate NER predictions for validation set"
    )
    parser.add_argument(
        "--validation",
        type=str,
        default=None,
        help="Path to validation.jsonl (default: auto-detect)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save validation_preds.jsonl (default: data/validation_preds.jsonl)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model directory (default: from config)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode for single sentence testing"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        # Interactive mode for single sentence testing
        config.validate_paths()
        test_sentence = input("Enter a sentence to test: ")
        symptoms = predict_symptoms(test_sentence, args.model)
        print("Symptoms detected:", symptoms)
    else:
        # Batch inference mode
        generate_validation_predictions(
            validation_path=args.validation,
            output_path=args.output,
            model_dir=args.model
        )
