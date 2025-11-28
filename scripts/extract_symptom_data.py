#!/usr/bin/env python3
"""
Script to extract symptom data from symptom_extractor.py and save to JSON files.

This script extracts the canonical symptom list and synonym mapping from the
symptom_extractor.py module and saves them as separate JSON files for easier
access and management.
"""

import json
import sys
import os

# Add the parent directory to the path to import from chatbot module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.symptom_extractor import symptom_list, synonym_map


def save_symptoms_to_json():
    """
    Extract and save the canonical symptom list to data/symptoms.json.
    
    The symptoms are saved as a simple list of strings, with each symptom
    representing a canonical medical symptom from the Symptom-severity.csv dataset.
    """
    print("Extracting canonical symptoms...")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save symptoms list
    symptoms_file = os.path.join(data_dir, 'symptoms.json')
    with open(symptoms_file, 'w', encoding='utf-8') as f:
        json.dump(symptom_list, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(symptom_list)} canonical symptoms to {symptoms_file}")


def save_synonyms_to_json():
    """
    Extract and save the synonym mapping to data/synonyms.json.
    
    The synonyms are saved as a dictionary mapping from synonym terms to
    their canonical symptom names. This allows for flexible symptom matching
    using common variations and alternative terms.
    """
    print("Extracting synonym mapping...")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save synonyms mapping
    synonyms_file = os.path.join(data_dir, 'synonyms.json')
    with open(synonyms_file, 'w', encoding='utf-8') as f:
        json.dump(synonym_map, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(synonym_map)} synonym mappings to {synonyms_file}")


def main():
    """
    Main function to extract and save all symptom data to JSON files.
    """
    print("Extracting symptom data from symptom_extractor.py...")
    print("=" * 60)
    
    try:
        # Extract and save symptoms
        save_symptoms_to_json()
        
        # Extract and save synonyms
        save_synonyms_to_json()
        
        print("=" * 60)
        print("Successfully extracted all symptom data!")
        print(f"Files created:")
        print(f"  - data/symptoms.json ({len(symptom_list)} symptoms)")
        print(f"  - data/synonyms.json ({len(synonym_map)} synonym mappings)")
        
    except Exception as e:
        print(f"Error extracting symptom data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
