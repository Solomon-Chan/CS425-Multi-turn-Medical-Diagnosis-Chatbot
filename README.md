# CS425 Multi-turn Medical Diagnosis Chatbot

A medical diagnosis chatbot that uses a fine-tuned BioBERT model for Named Entity Recognition (NER) to extract symptoms from patient conversations.

## Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Model Setup
The fine-tuned BioBERT model has been extracted to `models/biobert-ner/`. 
The model uses BIO tagging with labels: `B-SYM` (beginning), `I-SYM` (inside), `O` (outside).

## Usage

### Run the Chatbot
```bash
python chatbot/main.py
```

### Auto-labeling Scripts

#### Test auto-labeling with 200 samples
```bash
python scripts/auto_label.py -i data/ai-medical-chatbot.csv -d data --max-sentences 200 --fuzzy-cutoff 85
```

#### Auto-label all sentences
```bash
python scripts/auto_label.py -i data/ai-medical-chatbot.csv -d data --fuzzy-cutoff 85
```

#### Auto-label with validation examples
```bash
python scripts/auto_label.py --make-datasets --input data/ai-medical-chatbot.csv --data-dir data --min-per-class 900 --max-per-class 1000 --fuzzy-cutoff 90 --fuzzy-max-ngram 4
```

#### Test auto-labeling with generated data
```bash
python tests/test_auto_label_sample.py
```

### NER Inference

#### Test NER model inference
```bash
python scripts/infer_ner.py --model-dir "models/biobert_ner/checkpoint-biobert-ner-best" --text "Patient complains of chest pain and mild fever."
```

#### Evaluate NER model on validation set
```bash
python scripts/evaluate_ner.py --model-dir "models/biobert_ner/checkpoint-biobert-ner-best" --valid-path data/valid.jsonl
```

#### Test NER symptom extraction
```bash
python tests/test_infer_ner.py
```

- Quick test (recommended on CPU):
	Runs the evaluation on a limited number of examples so you can iterate quickly.
```bash
python scripts/evaluate_ner.py --max-examples 50
```

Notes:
- The evaluator now aligns gold labels to the model tokenizer before predicting, so subword boundaries are handled consistently.
- On CPU the full validation run may be slow; use `--max-examples` to test quickly.

If you want to reproduce Trainer-style evaluation (the same code used during fine-tuning), see `docs/EVALUATION.md` for a short guide.
## Project Structure

- `models/biobert-ner/`: Fine-tuned BioBERT NER model
- `scripts/`: Processing scripts (auto-labeling, evaluation, inference)
- `chatbot/`: Main chatbot application
- `data/`: Input data and auto-labeling outputs
- `bio_outputs/`: BIO-tagged training/validation data
- `config.py`: Centralized configuration for model paths and settings

See `PROJECT_STRUCTURE_EXPLANATION.md` for detailed documentation.

