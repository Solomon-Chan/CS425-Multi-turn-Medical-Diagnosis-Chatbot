"""
Medical Symptom Extractor (data-driven)

- Loads canonical symptoms and synonyms from data/symptoms.json and data/synonyms.json
- Provides canonicalization and high-level extraction (no auto-labeling here)
"""

import os
import re
import json
import csv
from typing import List, Dict, Tuple, Optional

# Get project root directory (parent of chatbot/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SEVERITY_CSV = os.path.join(DATA_DIR, "Symptom-severity.csv")


def load_symptom_data(data_dir: str = DATA_DIR) -> Tuple[List[str], Dict[str, str]]:
    """
    Load canonical symptoms and synonym mapping from JSON.
    """
    symptoms_path = os.path.join(data_dir, "symptoms.json")
    synonyms_path = os.path.join(data_dir, "synonyms.json")
    with open(symptoms_path, "r", encoding="utf-8") as f:
        symptom_list = json.load(f)
    with open(synonyms_path, "r", encoding="utf-8") as f:
        synonym_map = json.load(f)
    return symptom_list, synonym_map


def canonicalize_symptom(text: str) -> str:
    """
    Lowercase, remove punctuation except spaces/hyphens, collapse spaces.
    """
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"[^\w\s-]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_symptoms(
    text: str,
    symptom_list: Optional[List[str]] = None,
    synonym_map: Optional[Dict[str, str]] = None,
    data_dir: str = DATA_DIR,
) -> List[str]:
    """
    Return deduplicated canonical symptoms present in text via simple contains checks.
    """
    if not text:
        return []
    if symptom_list is None or synonym_map is None:
        symptom_list, synonym_map = load_symptom_data(data_dir)
    canonical_text = canonicalize_symptom(text)
    found = set()
    # exact canonical matches
    for s in symptom_list:
        if canonicalize_symptom(s) in canonical_text:
            found.add(s)
    # synonym matches
    for syn, canon in synonym_map.items():
        if canonicalize_symptom(syn) in canonical_text:
            found.add(canon)
    return sorted(found)


def get_symptom_severity(symptom: str) -> int:
    """
    Read severity weight for a canonical symptom from data/Symptom-severity.csv.
    """
    try:
        with open(SEVERITY_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                csv_symptom = row["Symptom"].replace("_", " ")
                if csv_symptom == symptom:
                    return int(row["weight"])
    except FileNotFoundError:
        print("Warning: Symptom-severity.csv not found")
    except Exception as e:
        print(f"Error reading severity: {e}")
    return 0


def get_all_symptoms_with_severity() -> Dict[str, int]:
    """
    Return dict of all symptom -> severity from data/Symptom-severity.csv.
    """
    out: Dict[str, int] = {}
    try:
        with open(SEVERITY_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Symptom"] != "prognosis":
                    out[row["Symptom"]] = int(row["weight"])
    except FileNotFoundError:
        print("Warning: Symptom-severity.csv not found")
    except Exception as e:
        print(f"Error reading severity table: {e}")
    return out
