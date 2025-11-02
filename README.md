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
python scripts/auto_label.py -i data/ai-medical-chatbot.csv -d data -m 2000 --validate -e 3
```

#### Test auto-labeling with generated data
```bash
python tests/test_auto_label_sample.py
```

### Create Stratified Validation Set
```bash
python scripts/select_gold_sample.py
```

### NER Inference

#### Test NER model inference
```bash
python scripts/infer_ner.py
```

#### Test NER symptom extraction
```bash
python tests/test_infer_ner.py
```

#### Evaluate NER model on validation set
```bash
python scripts/evaluate_ner.py
```

## Project Structure

- `models/biobert-ner/`: Fine-tuned BioBERT NER model
- `scripts/`: Processing scripts (auto-labeling, evaluation, inference)
- `chatbot/`: Main chatbot application
- `data/`: Input data and auto-labeling outputs
- `bio_outputs/`: BIO-tagged training/validation data
- `config.py`: Centralized configuration for model paths and settings

See `PROJECT_STRUCTURE_EXPLANATION.md` for detailed documentation.

