import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def load_model_and_tokenizer(model_path_or_name):
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    model = AutoModelForTokenClassification.from_pretrained(model_path_or_name)
    return tokenizer, model

def predict(text, tokenizer, model, label_map):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
    tokens = tokenizer.tokenize(text)
    
    # Convert predictions to labels
    labels = [label_map.get(pred, "O") for pred in predictions]
    
    # Combine token labels with text spans if needed
    return list(zip(tokens, labels))

if __name__ == "__main__":
    model_dir = "./biobert-ner"  # relative to your local repo path
    label_map = {0: "B-SYM", 1: "I-SYM", 2: "O"}
    tokenizer, model = load_model_and_tokenizer(model_dir)

    sample_text = "Patient complains of chest pain and mild fever."
    preds = predict(sample_text, tokenizer, model, label_map)
    for token, label in preds:
        print(f"{token:15s} {label}")
