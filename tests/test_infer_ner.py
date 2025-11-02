# scripts/test_infer_ner.py

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

LABEL_MAP = {0: "B-SYM", 1: "I-SYM", 2: "O"}  # Update if you use different labels

def extract_symptoms(tokens, preds, label_map):
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

def predict_symptoms(sentence, model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    tokens = tokenizer.tokenize(sentence)
    # If the model's output includes special tokens, strip them
    if len(preds) > len(tokens):
        offset = len(preds) - len(tokens)
        preds = preds[offset:]
    return extract_symptoms(tokens, preds, LABEL_MAP)

if __name__ == "__main__":
    # Example usage
    model_directory = "./biobert-ner"  # Path to your locally extracted model directory
    # test_sentence = "I have chest pain and a mild fever, but no headache or rash."
    test_sentence = input("Enter a sentence to test: ")
    symptoms = predict_symptoms(test_sentence, model_directory)
    print("Symptoms detected:", symptoms)
    # Output should look like: Symptoms detected: ['chest pain', 'mild fever']
