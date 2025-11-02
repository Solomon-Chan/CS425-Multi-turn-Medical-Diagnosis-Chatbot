import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import config

def load_model_and_tokenizer(model_path_or_name=None):
    """Load model and tokenizer from specified path or use config default."""
    if model_path_or_name is None:
        model_path_or_name = config.MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    model = AutoModelForTokenClassification.from_pretrained(model_path_or_name)
    return tokenizer, model

def predict(text, tokenizer, model, label_map=None):
    """Predict BIO tags for input text."""
    if label_map is None:
        label_map = config.LABEL_MAP
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=config.MAX_LENGTH)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
    tokens = tokenizer.tokenize(text)
    
    # Convert predictions to labels
    # Handle special tokens (CLS, SEP) by aligning with actual tokens
    # Model output format: [CLS] + token_predictions + [SEP]
    if len(predictions) > len(tokens):
        # Typically: len(predictions) = len(tokens) + 2 (CLS + SEP)
        if len(predictions) == len(tokens) + 2:
            # Remove CLS (first) and SEP (last) tokens
            predictions = predictions[1:-1]
        elif len(predictions) > len(tokens) + 2:
            # Handle padding or other cases
            # Remove CLS and SEP, keeping only token predictions
            predictions = predictions[1:-1]
        else:
            # Less common: just remove CLS if predictions are one longer
            predictions = predictions[1:]
    
    # Ensure predictions align with tokens
    # Trim if predictions are longer, pad if shorter
    if len(predictions) > len(tokens):
        predictions = predictions[:len(tokens)]
    elif len(predictions) < len(tokens):
        # Pad with "O" predictions if needed (shouldn't happen normally)
        predictions = predictions + [2] * (len(tokens) - len(predictions))  # 2 = O
    
    labels = [label_map.get(pred, "O") for pred in predictions]
    
    # Combine token labels with text spans if needed
    return list(zip(tokens, labels))

if __name__ == "__main__":
    # Validate paths exist
    config.validate_paths()
    
    tokenizer, model = load_model_and_tokenizer()
    model.eval()

    sample_text = "Patient complains of chest pain and mild fever."
    preds = predict(sample_text, tokenizer, model)
    print(f"Input: {sample_text}\n")
    print("Token-level predictions:")
    for token, label in preds:
        print(f"{token:20s} {label}")
