# Project Structure Explanation
## CS425 Multi-turn Medical Diagnosis Chatbot with Fine-tuned BioBERT NER

This document provides a comprehensive explanation of the entire project structure, including the purpose and functionality of each directory and file.

---

## 📋 Project Overview

This project is a **Medical Diagnosis Chatbot** that uses a fine-tuned BioBERT model for **Named Entity Recognition (NER)** to extract symptoms from patient conversations. The system uses BIO (Beginning-Inside-Outside) tagging to identify symptom mentions in medical text.

**Core Workflow:**
1. **Data Processing**: Auto-label medical conversations using symptom dictionaries
2. **Model Training**: Fine-tune BioBERT on symptom-labeled data (BIO tags: B-SYM, I-SYM, O)
3. **Inference**: Use the fine-tuned model to extract symptoms from new patient queries
4. **Chatbot Integration**: Extract symptoms for medical diagnosis conversations

---

## 📁 Directory Structure & Component Analysis

### **Root Directory Files**

#### `.gitignore`
- **Purpose**: Excludes files from version control (cache, models, data files, tokenizers, etc.)
- **Key exclusions**: `__pycache__/`, model checkpoints (`*.pt`, `*.bin`, `*.ckpt`), data files (`*.csv`, `*.jsonl`), tokenizer files, Colab Drive mounts

#### `README.md`
- **Purpose**: Quick reference guide with commands for common operations
- **Content**: Installation commands, auto-labeling commands, gold set creation, NER testing

#### `requirements.txt`
- **Status**: Currently empty - needs to be populated with dependencies
- **Should contain**: `transformers`, `torch`, `pandas`, `rapidfuzz`, `seqeval`, `numpy`, etc.

---

### **📂 `/scripts` - Core Processing Scripts**

Contains all executable Python scripts for data processing, model inference, and evaluation.

#### `scripts/auto_label.py` (472 lines)
- **Purpose**: Automated symptom labeling system that extracts symptom spans from medical conversations
- **Key Functionality**:
  - Processes CSV files containing medical chatbot conversations
  - Extracts symptoms from "Description" and "Patient" columns (skips "Doctor" column)
  - Uses **exact matching** first, then **fuzzy matching** as fallback
  - Applies **negation/uncertainty filters** (NegEx-style) to avoid false positives
  - Uses **headword gating** for fuzzy matches (requires medical headwords like "pain", "fever", "rash")
  - Outputs character-level spans: `(start, end, canonical_label, confidence)`
- **Outputs**:
  - `data/auto_labeled_sample.jsonl`: Sentences with symptom spans found
  - `data/unknown_spans.jsonl`: Sentences without symptoms (with suggestions)
  - `data/label_counts.csv`: Per-symptom span and sentence counts
- **Usage**: `python scripts/auto_label.py -i data/ai-medical-chatbot.csv -d data --max-sentences 200 --fuzzy-cutoff 85`

#### `scripts/select_gold_sample.py` (198 lines)
- **Purpose**: Creates a stratified validation set from auto-labeled BIO data
- **Key Functionality**:
  - Selects samples with balanced representation:
    - 15% negative examples (no symptoms)
    - 30% ambiguous cases (multiple symptoms or negation)
    - 35% rare label coverage (ensures low-frequency symptoms are represented)
    - Remaining from common cases
  - Uses auto-labeled sample to identify canonical symptom labels
  - Adds metadata: `num_entities`, `has_negation`, `labels_from_silver`
- **Output**: `data/validation_sample.jsonl` (default: 150 examples)
- **Usage**: `python scripts/select_gold_sample.py`

#### `scripts/infer_ner.py` (32 lines)
- **Purpose**: Load and run inference with the fine-tuned BioBERT NER model
- **Key Functionality**:
  - Loads model and tokenizer from specified directory
  - Performs token-level prediction on input text
  - Maps predictions to BIO labels: `{0: "B-SYM", 1: "I-SYM", 2: "O"}`
  - Returns token-label pairs
- **Current Issue**: Hardcoded path `"./biobert-ner"` - needs to point to actual model checkpoint
- **Usage**: `python scripts/infer_ner.py` (currently has hardcoded test example)

#### `scripts/evaluate_ner.py` (74 lines)
- **Purpose**: Evaluate NER model performance on validation set
- **Key Functionality**:
  - Loads model and validation data
  - Predicts tags for each example
  - Computes metrics using `seqeval`: precision, recall, F1-score, classification report
  - Handles token alignment issues (subword tokenization mismatches)
- **Output**: `reports/ner_eval.txt` (if reports/ directory exists)
- **Current Issue**: Hardcoded path `"./biobert-ner"` - needs model path configuration
- **Usage**: `python scripts/evaluate_ner.py`

---

### **📂 `/chatbot` - Chatbot Application**

Contains the main chatbot application that integrates symptom extraction.

#### `chatbot/symptom_extractor.py` (105 lines)
- **Purpose**: Core symptom extraction utilities used by both scripts and chatbot
- **Key Functions**:
  - `load_symptom_data()`: Loads `symptoms.json` and `synonyms.json` from data directory
  - `canonicalize_symptom()`: Normalizes symptom text (lowercase, remove punctuation)
  - `extract_symptoms()`: Simple contains-based symptom extraction (not using NER model)
  - `get_symptom_severity()`: Reads severity weights from CSV (for prognosis/diagnosis)
- **Note**: This is currently rule-based extraction; should integrate with NER model

#### `chatbot/main.py`
- **Status**: Empty file
- **Expected Purpose**: Main chatbot application entry point
- **Should integrate**: NER model inference for real-time symptom extraction

---

### **📂 `/tests` - Test Suite**

Contains unit tests and validation scripts.

#### `tests/test_infer_ner.py` (49 lines)
- **Purpose**: Test script for NER inference functionality
- **Key Functionality**:
  - Tests symptom extraction from model predictions
  - Groups consecutive B-SYM/I-SYM tokens into full symptom phrases
  - Handles special token offsets
  - Interactive: prompts user for test sentence
- **Current Issue**: Hardcoded path `"./biobert-ner"`
- **Usage**: `python tests/test_infer_ner.py`

#### `tests/test_bio_tags.py` (250 lines)
- **Purpose**: Comprehensive validation of BIO tagging accuracy
- **Key Functionality**:
  - Validates positive examples (sentences with symptoms) - checks if BIO tags match expected spans
  - Validates negative examples (sentences without symptoms) - checks if all tags are "O"
  - Computes token-level and span-level accuracy metrics
  - Per-class precision/recall/F1 for B-SYM, I-SYM, O
  - Uses BioBERT tokenizer to align tokens with character spans
- **Usage**: `python tests/test_bio_tags.py`

#### `tests/test_auto_label_sample.py` (103 lines)
- **Purpose**: End-to-end test of auto-labeling pipeline
- **Key Functionality**:
  - Creates or finds sample CSV
  - Runs auto-labeling on small sample (150 sentences)
  - Validates span integrity (bounds check, canonical label check)
  - Prints sample outputs for manual inspection
- **Usage**: `python tests/test_auto_label_sample.py`

---

### **📂 `/data` - Data Files**

Contains input data, symptom dictionaries, and generated outputs.

#### **Input Data:**
- `ai-medical-chatbot.csv`: Source medical conversation dataset (contains Description, Patient, Doctor columns)
- `symptoms.json`: List of canonical symptom names (137 symptoms)
- `synonyms.json`: Mapping of symptom synonyms/variants to canonical names
- `symptoms.csv`: Alternative format symptom list (if exists)

#### **Generated/Processed Data:**
- `auto_labeled_sample.jsonl`: Output from auto-labeling - sentences with symptom spans
  - Format: `{"text": "...", "spans": [(start, end, label, conf), ...], "source_column": "...", "row_index": ...}`
- `unknown_spans.jsonl`: Sentences without detected symptoms (with fuzzy match suggestions)
- `label_counts.csv`: Statistics showing frequency of each symptom label
- `validation_sample.jsonl`: Stratified gold set for evaluation (150 examples with BIO tags)

---

### **📂 `/bio_outputs` - BIO Tagging Outputs**

Contains the tokenized and BIO-tagged training/validation data.

#### **Files:**
- `train.jsonl`: Full training set with BIO tags
  - Format: `{"text": "...", "tokens": ["hi", "doctor", ...], "tags": ["O", "O", "B-SYM", "I-SYM", ...]}`
- `validation.jsonl`: Validation set with BIO tags
- `validation_sample.jsonl`: Stratified validation sample (same as `data/validation_sample.jsonl` - possible duplicate)
- `train_conll.txt`: CoNLL format training data (alternative format)
- `auto_labeled_sample.jsonl`: Same as data directory - possible duplicate
- `unknown_spans.jsonl`: Same as data directory - possible duplicate
- `tokenizer_biobert_v1.1/`: BioBERT tokenizer files (vocab.txt, tokenizer.json, etc.)

**Note**: There is overlap between `/data` and `/bio_outputs`. The structure should be consolidated.

---

### **📂 `/model_checkpoint` - Fine-tuned Model**

Contains the fine-tuned BioBERT model weights and configuration.

#### **Structure:**
```
model_checkpoint/
└── content/
    └── drive/
        └── MyDrive/
            └── Colab Notebooks/
                └── CS425 GenAI for NLC/
                    ├── checkpoint-37503/          # ← ACTUAL MODEL CHECKPOINT
                    │   ├── config.json           # Model configuration (BIO labels: B-SYM, I-SYM, O)
                    │   ├── model.safetensors     # Model weights
                    │   ├── tokenizer.json        # Tokenizer configuration
                    │   ├── tokenizer_config.json
                    │   ├── special_tokens_map.json
                    │   ├── vocab.txt             # Vocabulary
                    │   ├── trainer_state.json    # Training metadata
                    │   ├── training_args.bin     # Training arguments
                    │   └── [optimizer/scheduler files] # Not needed for inference
                    ├── config.json               # Root-level config (duplicate)
                    ├── model.safetensors          # Root-level weights (may be duplicate)
                    ├── tokenizer_biobert_v1.1/    # Tokenizer directory
                    └── [other training files]
```

#### **Key Model Information:**
- **Architecture**: `BertForTokenClassification`
- **Base Model**: BioBERT (dmis-lab/biobert-base-cased-v1.1)
- **Labels**: 3 classes - `{0: "B-SYM", 1: "I-SYM", 2: "O"}`
- **Checkpoint**: `checkpoint-37503` is the trained model (37,503 training steps)

**⚠️ Current Issue**: The model checkpoint is deeply nested (Colab Drive structure). Scripts reference `"./biobert-ner"` which doesn't exist. The model needs to be extracted to a cleaner location.

---

### **📂 `/models` - Model Storage (Empty)**

- **Purpose**: Intended for storing model files
- **Status**: Currently empty
- **Recommendation**: Should contain the extracted fine-tuned model for easy access

---

### **📂 `/notebooks` - Jupyter Notebooks**

- `CS425_Project.ipynb`: Research/experimentation notebook (may contain training code or analysis)

---

## 🔧 Integration Issues & Recommendations

### **Current Problems:**

1. **Model Path Mismatch**
   - Scripts reference `"./biobert-ner"` but model is in deeply nested `model_checkpoint/content/drive/.../checkpoint-37503/`
   - Solution: Extract model to `models/biobert-ner/` or update all scripts to use correct path

2. **Duplicate Files**
   - Files exist in both `/data` and `/bio_outputs` (e.g., `auto_labeled_sample.jsonl`, `validation_sample.jsonl`)
   - Solution: Consolidate to single source of truth

3. **Missing Requirements**
   - `requirements.txt` is empty
   - Solution: Add all dependencies

4. **Empty Chatbot Main**
   - `chatbot/main.py` is empty
   - Solution: Implement chatbot with NER integration

5. **Inconsistent Model Loading**
   - Some scripts use hardcoded paths, others use relative paths
   - Solution: Use environment variable or config file for model path

---

### **Recommended Cleanup Actions:**

1. **Extract Model Checkpoint**
   - Copy `model_checkpoint/.../checkpoint-37503/*` → `models/biobert-ner/`
   - Keep only essential files: `config.json`, `model.safetensors`, tokenizer files, `vocab.txt`
   - Remove training artifacts (optimizer.pt, scheduler.pt, etc.)

2. **Consolidate Data Directories**
   - Keep source data in `/data`
   - Keep processed BIO outputs in `/bio_outputs`
   - Remove duplicates

3. **Update Scripts**
   - Create `config.py` with `MODEL_PATH = "models/biobert-ner"`
   - Update all scripts to import and use this path

4. **Populate Requirements**
   - Add: `transformers>=4.57.1`, `torch`, `pandas`, `rapidfuzz`, `seqeval`, `numpy`

5. **Complete Chatbot Integration**
   - Implement `chatbot/main.py` using `infer_ner.py` for symptom extraction

---

## 🔄 Data Flow Summary

```
1. Source Data (CSV)
   ↓
2. Auto-labeling (auto_label.py)
   → Character spans with symptom labels
   ↓
3. Tokenization & BIO Tagging
   → train.jsonl, validation.jsonl (token-level BIO tags)
   ↓
4. Model Training (Colab Notebook - not in repo)
   → Fine-tuned BioBERT checkpoint
   ↓
5. Model Inference (infer_ner.py)
   → Extract symptoms from new patient queries
   ↓
6. Chatbot Integration (main.py - TODO)
   → Real-time symptom extraction for diagnosis
```

---

## 📝 File Summary Table

| File/Folder | Purpose | Status |
|------------|---------|--------|
| `scripts/auto_label.py` | Auto-label symptoms from CSV | ✅ Complete |
| `scripts/select_gold_sample.py` | Create stratified validation set | ✅ Complete |
| `scripts/infer_ner.py` | Load and infer with NER model | ⚠️ Needs path fix |
| `scripts/evaluate_ner.py` | Evaluate model on validation set | ⚠️ Needs path fix |
| `chatbot/symptom_extractor.py` | Symptom extraction utilities | ✅ Complete |
| `chatbot/main.py` | Main chatbot app | ❌ Empty |
| `tests/test_*.py` | Test suite | ✅ Complete (needs path fixes) |
| `model_checkpoint/` | Fine-tuned model | ⚠️ Needs extraction |
| `data/` | Input/output data | ⚠️ Has duplicates |
| `bio_outputs/` | BIO-tagged data | ⚠️ Has duplicates |
| `requirements.txt` | Dependencies | ❌ Empty |

---

This structure analysis provides a complete understanding of your project. The next step would be to implement the cleanup recommendations and integrate the NER model properly into the chatbot application.

