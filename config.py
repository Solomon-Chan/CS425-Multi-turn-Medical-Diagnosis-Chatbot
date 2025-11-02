"""
Configuration file for CS425 Medical Diagnosis Chatbot
Centralizes all paths and settings for easy maintenance.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Model paths
MODEL_DIR = PROJECT_ROOT / "models" / "biobert-ner"
MODEL_PATH = str(MODEL_DIR)

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
BIO_OUTPUTS_DIR = PROJECT_ROOT / "bio_outputs"

# Label mapping for BIO tagging
LABEL_MAP = {
    0: "B-SYM",  # Beginning of symptom
    1: "I-SYM",  # Inside symptom
    2: "O"       # Outside (not a symptom)
}

# Inverse label mapping
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
LABEL2ID = LABEL_MAP

# Model configuration
MAX_LENGTH = 128  # Maximum sequence length for tokenization
TRUNCATION = True

# Path validation
def validate_paths():
    """Validate that required paths exist."""
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}\n"
            f"Please ensure the fine-tuned model is extracted to {MODEL_DIR}"
        )
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    return True

