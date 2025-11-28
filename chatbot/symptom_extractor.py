"""
Medical Symptom Extractor Module

This module provides comprehensive functionality for extracting and processing medical symptoms
from text data. It includes canonical symptom lists, synonym mapping, character span extraction,
and auto-labeling capabilities for weak supervision in medical NLP tasks.

Key Features:
- 133 canonical symptoms from medical dataset
- Comprehensive synonym mapping for flexible matching
- Character span extraction with longest-first matching
- Auto-labeling for weak supervision
- Severity weight mapping from CSV data

Author: CS425 Medical Diagnosis Chatbot Team
"""

import re
import csv
import json
import pandas as pd
from typing import List, Dict, Set, Tuple

# =============================================================================
# CANONICAL SYMPTOM DATA
# =============================================================================

# Canonical symptom list extracted from Symptom-severity.csv
# These 133 symptoms represent the standardized medical symptoms used in the dataset
# Each symptom is converted from underscore format (e.g., "joint_pain") to readable format (e.g., "joint pain")
symptom_list = [
    "itching", "skin rash", "nodal skin eruptions", "continuous sneezing", "shivering",
    "chills", "joint pain", "stomach pain", "acidity", "ulcers on tongue", "muscle wasting",
    "vomiting", "burning micturition", "spotting urination", "fatigue", "weight gain",
    "anxiety", "cold hands and feets", "mood swings", "weight loss", "restlessness",
    "lethargy", "patches in throat", "irregular sugar level", "cough", "high fever",
    "sunken eyes", "breathlessness", "sweating", "dehydration", "indigestion", "headache",
    "yellowish skin", "dark urine", "nausea", "loss of appetite", "pain behind the eyes",
    "back pain", "constipation", "abdominal pain", "diarrhoea", "mild fever", "yellow urine",
    "yellowing of eyes", "acute liver failure", "fluid overload", "swelling of stomach",
    "swelled lymph nodes", "malaise", "blurred and distorted vision", "phlegm",
    "throat irritation", "redness of eyes", "sinus pressure", "runny nose", "congestion",
    "chest pain", "weakness in limbs", "fast heart rate", "pain during bowel movements",
    "pain in anal region", "bloody stool", "irritation in anus", "neck pain", "dizziness",
    "cramps", "bruising", "obesity", "swollen legs", "swollen blood vessels",
    "puffy face and eyes", "enlarged thyroid", "brittle nails", "swollen extremeties",
    "excessive hunger", "extra marital contacts", "drying and tingling lips", "slurred speech",
    "knee pain", "hip joint pain", "muscle weakness", "stiff neck", "swelling joints",
    "movement stiffness", "spinning movements", "loss of balance", "unsteadiness",
    "weakness of one body side", "loss of smell", "bladder discomfort", "foul smell of urine",
    "continuous feel of urine", "passage of gases", "internal itching", "toxic look",
    "depression", "irritability", "muscle pain", "altered sensorium", "red spots over body",
    "belly pain", "abnormal menstruation", "dischromic patches", "watering from eyes",
    "increased appetite", "polyuria", "family history", "mucoid sputum", "rusty sputum",
    "lack of concentration", "visual disturbances", "receiving blood transfusion",
    "receiving unsterile injections", "coma", "stomach bleeding", "distention of abdomen",
    "history of alcohol consumption", "blood in sputum", "prominent veins on calf",
    "palpitations", "painful walking", "pus filled pimples", "blackheads", "scurring",
    "skin peeling", "silver like dusting", "small dents in nails", "inflammatory nails",
    "blister", "red sore around nose", "yellow crust ooze"
]

# =============================================================================
# SYNONYM MAPPING
# =============================================================================

# Synonym mapping for common symptom variations
# This dictionary maps alternative terms and common variations to their canonical symptom names
# This enables flexible matching of symptoms using everyday language and medical terminology
# Format: "synonym_term": "canonical_symptom_name"
synonym_map = {
    # Fever variations
    "fever": "high fever",
    "feverish": "high fever", 
    "temperature": "high fever",
    "hot": "high fever",
    "mild fever": "mild fever",
    "low fever": "mild fever",
    
    # Pain variations
    "pain": "joint pain",
    "ache": "joint pain",
    "hurts": "joint pain",
    "sore": "joint pain",
    "stomach ache": "stomach pain",
    "belly ache": "stomach pain",
    "abdominal ache": "abdominal pain",
    "back ache": "back pain",
    "neck ache": "neck pain",
    "knee ache": "knee pain",
    "hip ache": "hip joint pain",
    "chest ache": "chest pain",
    "head ache": "headache",
    "migraine": "headache",
    
    # Breathing/respiratory
    "breathing problems": "breathlessness",
    "shortness of breath": "breathlessness",
    "can't breathe": "breathlessness",
    "wheezing": "breathlessness",
    "coughing": "cough",
    "hacking": "cough",
    "sneezing": "continuous sneezing",
    "runny nose": "runny nose",
    "stuffy nose": "congestion",
    "nasal congestion": "congestion",
    "blocked nose": "congestion",
    
    # Skin conditions
    "rash": "skin rash",
    "skin irritation": "skin rash",
    "red skin": "skin rash",
    "itchy skin": "itching",
    "itch": "itching",
    "scratching": "itching",
    "bruise": "bruising",
    "bruised": "bruising",
    "black and blue": "bruising",
    
    # Digestive
    "throwing up": "vomiting",
    "puking": "vomiting",
    "sick": "vomiting",
    "nauseous": "nausea",
    "queasy": "nausea",
    "diarrhea": "diarrhoea",
    "loose stools": "diarrhoea",
    "constipated": "constipation",
    "can't go": "constipation",
    "indigestion": "indigestion",
    "upset stomach": "indigestion",
    "heartburn": "acidity",
    "acid reflux": "acidity",
    
    # Fatigue/energy
    "tired": "fatigue",
    "exhausted": "fatigue",
    "weak": "fatigue",
    "no energy": "fatigue",
    "lethargic": "lethargy",
    "sluggish": "lethargy",
    "drained": "fatigue",
    
    # Mental/emotional
    "anxious": "anxiety",
    "worried": "anxiety",
    "nervous": "anxiety",
    "sad": "depression",
    "down": "depression",
    "moody": "mood swings",
    "irritable": "irritability",
    "cranky": "irritability",
    "restless": "restlessness",
    "can't sit still": "restlessness",
    
    # Urinary
    "peeing": "burning micturition",
    "urination": "burning micturition",
    "urinating": "burning micturition",
    "bladder pain": "bladder discomfort",
    "urine smell": "foul smell of urine",
    "smelly urine": "foul smell of urine",
    
    # Vision/eyes
    "blurry vision": "blurred and distorted vision",
    "can't see clearly": "blurred and distorted vision",
    "red eyes": "redness of eyes",
    "pink eye": "redness of eyes",
    "watery eyes": "watering from eyes",
    "tearing": "watering from eyes",
    
    # Swelling
    "swollen": "swelling joints",
    "puffy": "swelling joints",
    "inflamed": "swelling joints",
    "swollen legs": "swollen legs",
    "puffy legs": "swollen legs",
    "swollen face": "puffy face and eyes",
    "puffy face": "puffy face and eyes",
    
    # Other common variations
    "dizzy": "dizziness",
    "lightheaded": "dizziness",
    "spinning": "spinning movements",
    "vertigo": "spinning movements",
    "unsteady": "unsteadiness",
    "wobbly": "unsteadiness",
    "balance problems": "loss of balance",
    "can't balance": "loss of balance",
    "muscle cramps": "cramps",
    "spasms": "cramps",
    "shaking": "shivering",
    "trembling": "shivering",
    "chilly": "chills",
    "cold": "chills",
    "sweaty": "sweating",
    "perspiring": "sweating",
    "thirsty": "dehydration",
    "dehydrated": "dehydration",
    "hungry": "excessive hunger",
    "starving": "excessive hunger",
    "no appetite": "loss of appetite",
    "not hungry": "loss of appetite",
    "weight increase": "weight gain",
    "gained weight": "weight gain",
    "weight decrease": "weight loss",
    "lost weight": "weight loss",
    "throat pain": "throat irritation",
    "sore throat": "throat irritation",
    "scratchy throat": "throat irritation",
    "phlegm": "phlegm",
    "mucus": "phlegm",
    "sputum": "phlegm"
}

# =============================================================================
# CORE EXTRACTION FUNCTIONS
# =============================================================================

def canonicalize_symptom(symptom: str) -> str:
    """
    Canonicalize a symptom by converting to lowercase and removing punctuation.
    
    This function normalizes symptom text for consistent matching by:
    - Converting to lowercase for case-insensitive matching
    - Removing punctuation (except spaces and hyphens) to handle variations
    - Normalizing whitespace to handle multiple spaces
    - Stripping leading/trailing whitespace for clean text
    
    Args:
        symptom: Raw symptom string (e.g., "High Fever!", "joint  pain")
        
    Returns:
        Canonicalized symptom string (e.g., "high fever", "joint pain")
    """
    # Convert to lowercase for case-insensitive matching
    canonical = symptom.lower()
    
    # Remove punctuation except spaces and hyphens to handle variations like "High Fever!"
    canonical = re.sub(r'[^\w\s-]', '', canonical)
    
    # Replace multiple spaces with single space to normalize whitespace
    canonical = re.sub(r'\s+', ' ', canonical)
    
    # Strip leading/trailing whitespace for clean text
    canonical = canonical.strip()
    
    return canonical


def extract_symptoms(text: str) -> List[str]:
    """
    Extract symptoms from input text using canonical symptom list and synonym mapping.
    
    This function performs symptom extraction by:
    1. Canonicalizing the input text for consistent matching
    2. Checking for exact matches with canonical symptoms
    3. Checking for matches with synonym terms
    4. Returning a deduplicated list of canonical symptoms
    
    Args:
        text: Input text to extract symptoms from (e.g., "I have a fever and headache")
        
    Returns:
        List of canonical symptoms found in the text (e.g., ["high fever", "headache"])
    """
    if not text:
        return []
    
    # Canonicalize the input text for consistent matching
    canonical_text = canonicalize_symptom(text)
    
    found_symptoms = set()  # Use set to avoid duplicates
    
    # First, check for exact matches with canonical symptoms
    # This ensures we catch direct matches with our standardized symptom names
    for symptom in symptom_list:
        canonical_symptom = canonicalize_symptom(symptom)
        if canonical_symptom in canonical_text:
            found_symptoms.add(symptom)
    
    # Then check for synonym matches
    # This allows us to catch variations and common terms that map to canonical symptoms
    for synonym, canonical_symptom in synonym_map.items():
        canonical_synonym = canonicalize_symptom(synonym)
        if canonical_synonym in canonical_text:
            found_symptoms.add(canonical_symptom)
    
    return list(found_symptoms)


# =============================================================================
# SEVERITY MAPPING FUNCTIONS
# =============================================================================

def get_symptom_severity(symptom: str) -> int:
    """
    Get the severity weight for a given symptom from the CSV data.
    
    This function reads the Symptom-severity.csv file to retrieve the severity weight
    (1-7 scale) for a given canonical symptom. The severity weights are used to
    prioritize symptoms based on their medical importance.
    
    Args:
        symptom: Canonical symptom name (e.g., "high fever", "joint pain")
        
    Returns:
        Severity weight (1-7) where 7 is most severe, or 0 if not found
    """
    # Read the CSV file to get severity weights
    try:
        with open('data/Symptom-severity.csv', 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Convert CSV symptom to match our format (replace underscores with spaces)
                # CSV format: "joint_pain" -> our format: "joint pain"
                csv_symptom = row['Symptom'].replace('_', ' ')
                if csv_symptom == symptom:
                    return int(row['weight'])
    except FileNotFoundError:
        print("Warning: Symptom-severity.csv not found")
    except Exception as e:
        print(f"Error reading symptom severity: {e}")
    
    return 0  # Return 0 if symptom not found or error occurred


def get_all_symptoms_with_severity() -> Dict[str, int]:
    """
    Get all symptoms with their severity weights.
    
    This function reads the entire Symptom-severity.csv file and returns a dictionary
    mapping all symptom names to their severity weights. This is useful for bulk
    processing or analysis of symptom severity distributions.
    
    Returns:
        Dictionary mapping symptom names to severity weights
        Format: {"joint_pain": 3, "high_fever": 7, ...}
    """
    symptoms_with_severity = {}
    
    try:
        with open('data/Symptom-severity.csv', 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Skip the prognosis row as it's not a symptom
                if row['Symptom'] != 'prognosis':
                    symptoms_with_severity[row['Symptom']] = int(row['weight'])
    except FileNotFoundError:
        print("Warning: Symptom-severity.csv not found")
    except Exception as e:
        print(f"Error reading symptom severity: {e}")
    
    return symptoms_with_severity


# =============================================================================
# CHARACTER SPAN EXTRACTION (FOR WEAK SUPERVISION)
# =============================================================================

def extract_character_spans(text: str) -> List[Tuple[int, int, str]]:
    """
    Extract character spans for symptoms using longest-first substring matching.
    
    This function is designed for weak supervision and provides precise character-level
    spans for symptoms in text. It uses a sophisticated matching algorithm that:
    1. Creates a comprehensive dictionary of all phrases to match (symptoms + synonyms)
    2. Sorts phrases by length (longest first) to prevent short matches overshadowing long ones
    3. Uses word boundary detection to ensure complete word matches
    4. Tracks used character positions to avoid overlapping matches
    5. Returns precise character spans for tokenizer compatibility
    
    Args:
        text: Input text to extract spans from (e.g., "I have a fever and headache")
        
    Returns:
        List of (start_char, end_char, canonical_label) tuples
        Example: [(9, 14, 'high fever'), (19, 27, 'headache')]
    """
    if not text:
        return []
    
    spans = []
    
    # Create a comprehensive dictionary of all phrases to match
    # Include both canonical symptoms and synonyms for complete coverage
    all_phrases = {}
    
    # Add canonical symptoms to the matching dictionary
    for symptom in symptom_list:
        canonical_symptom = canonicalize_symptom(symptom)
        all_phrases[canonical_symptom] = symptom
    
    # Add synonyms to the matching dictionary
    for synonym, canonical_symptom in synonym_map.items():
        canonical_synonym = canonicalize_symptom(synonym)
        all_phrases[canonical_synonym] = canonical_symptom
    
    # Sort phrases by length (longest first) to prevent short matches overshadowing long ones
    # This ensures "high fever" is matched before "fever" to avoid partial matches
    sorted_phrases = sorted(all_phrases.items(), key=lambda x: len(x[0]), reverse=True)
    
    # Track used character positions to avoid overlapping matches
    used_positions = set()
    
    # Find all matches using longest-first approach
    for phrase, canonical_label in sorted_phrases:
        if not phrase or len(phrase) < 3:  # Skip empty or very short phrases
            continue
            
        # Search in the original text (case-insensitive)
        text_lower = text.lower()
        start = 0
        
        while True:
            # Find the next occurrence of the phrase in lowercase text
            pos = text_lower.find(phrase, start)
            if pos == -1:
                break
            
            # Check if this is a word boundary match (not part of a larger word)
            # This prevents matching "pain" inside "painful" or "paint"
            is_word_boundary = True
            
            # Check character before the match
            if pos > 0 and text[pos-1].isalnum():
                is_word_boundary = False
            
            # Check character after the match
            end_pos = pos + len(phrase)
            if end_pos < len(text) and text[end_pos].isalnum():
                is_word_boundary = False
            
            # Only add if it's a word boundary match and doesn't overlap with existing spans
            if is_word_boundary:
                phrase_positions = set(range(pos, end_pos))
                
                if not phrase_positions.intersection(used_positions):
                    # No overlap, add this span
                    spans.append((pos, end_pos, canonical_label))
                    used_positions.update(phrase_positions)
            
            # Move to next potential match
            start = pos + 1
    
    # Sort spans by start position for consistent ordering
    spans.sort(key=lambda x: x[0])
    
    return spans


# =============================================================================
# AUTO-LABELING FUNCTIONS (FOR WEAK SUPERVISION)
# =============================================================================

def auto_label_sentences(csv_file: str, output_file: str, max_sentences: int = 200) -> None:
    """
    Auto-label sentences from the medical chatbot dataset with character spans.
    
    This function processes the medical chatbot dataset for weak supervision by:
    1. Reading the CSV file in chunks to handle large files efficiently
    2. Processing each text column (Description, Patient, Doctor)
    3. Extracting character spans for symptoms in each sentence
    4. Saving results in JSONL format for easy processing
    5. Providing statistics on the labeling process
    
    Args:
        csv_file: Path to the input CSV file (e.g., 'data/ai-medical-chatbot.csv')
        output_file: Path to the output JSONL file (e.g., 'data/auto_labeled.jsonl')
        max_sentences: Maximum number of sentences to process (default: 200)
    """
    print(f"Loading dataset from {csv_file}...")
    
    # Read the CSV file in chunks to handle large files efficiently
    chunk_size = 1000
    processed_sentences = 0
    labeled_data = []
    
    try:
        # Read CSV in chunks to handle large files without loading everything into memory
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            if processed_sentences >= max_sentences:
                break
                
            for _, row in chunk.iterrows():
                if processed_sentences >= max_sentences:
                    break
                
                # Process each column that contains text
                # The medical chatbot dataset has three main text columns
                for column in ['Description', 'Patient', 'Doctor']:
                    if column in row and pd.notna(row[column]):
                        text = str(row[column]).strip()
                        # Skip very short texts as they're unlikely to contain meaningful symptoms
                        if text and len(text) > 10:
                            # Extract character spans for symptoms in this text
                            spans = extract_character_spans(text)
                            
                            # Create labeled entry with metadata
                            labeled_entry = {
                                'text': text,
                                'spans': spans,
                                'source_column': column,
                                'row_index': row.name if hasattr(row, 'name') else processed_sentences
                            }
                            
                            labeled_data.append(labeled_entry)
                            processed_sentences += 1
                            
                            if processed_sentences >= max_sentences:
                                break
                
                if processed_sentences >= max_sentences:
                    break
                    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    print(f"Processed {processed_sentences} sentences")
    print(f"Saving labeled data to {output_file}...")
    
    # Save to JSONL format (one JSON object per line)
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in labeled_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Successfully saved {len(labeled_data)} labeled sentences to {output_file}")
        
        # Print statistics to help evaluate the labeling quality
        total_spans = sum(len(entry['spans']) for entry in labeled_data)
        sentences_with_spans = sum(1 for entry in labeled_data if entry['spans'])
        
        print(f"Statistics:")
        print(f"  Total sentences: {len(labeled_data)}")
        print(f"  Sentences with spans: {sentences_with_spans}")
        print(f"  Total spans: {total_spans}")
        print(f"  Average spans per sentence: {total_spans/len(labeled_data):.2f}")
        
    except Exception as e:
        print(f"Error saving to JSONL file: {e}")


def validate_spans(jsonl_file: str, num_examples: int = 5) -> None:
    """
    Validate character spans by showing examples with highlighted spans.
    
    This function helps verify the quality of the auto-labeling by:
    1. Reading the JSONL file with labeled data
    2. Showing examples with spans highlighted in the original text
    3. Displaying the character positions and canonical labels
    4. Providing visual feedback on the extraction quality
    
    Args:
        jsonl_file: Path to the JSONL file with labeled data
        num_examples: Number of examples to show (default: 5)
    """
    print(f"\nValidating spans from {jsonl_file}...")
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            examples_shown = 0
            
            for line in f:
                if examples_shown >= num_examples:
                    break
                    
                entry = json.loads(line.strip())
                text = entry['text']
                spans = entry['spans']
                
                # Only show examples that have spans (symptoms found)
                if spans:
                    print(f"\nExample {examples_shown + 1}:")
                    print(f"Text: {text}")
                    print(f"Spans: {spans}")
                    
                    # Show highlighted text with spans marked
                    highlighted = text
                    # Sort spans by start position in reverse order to avoid index shifting
                    # when inserting highlight markers
                    sorted_spans = sorted(spans, key=lambda x: x[0], reverse=True)
                    
                    for start, end, label in sorted_spans:
                        span_text = text[start:end]
                        # Insert highlight markers around the span
                        highlighted = highlighted[:start] + f"[{span_text}]({label})" + highlighted[end:]
                    
                    print(f"Highlighted: {highlighted}")
                    examples_shown += 1
                    
    except Exception as e:
        print(f"Error validating spans: {e}")


# =============================================================================
# MAIN EXECUTION AND TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Main execution block for testing the symptom extractor functionality.
    
    This section provides interactive testing of the symptom extraction capabilities,
    including basic symptom extraction, severity mapping, character span extraction,
    and auto-labeling functionality.
    """
    # Test the symptom extractor with user input
    print("Medical Symptom Extractor - Interactive Testing")
    print("=" * 50)
    test_text = input("Enter text to analyze for symptoms: ")

    print(f"\nTest text: {test_text}")
    print(f"Extracted symptoms: {extract_symptoms(test_text)}")
    
    # Test individual symptoms with severity mapping
    print("\nSymptom Analysis:")
    for symptom in extract_symptoms(test_text):
        severity = get_symptom_severity(symptom)
        print(f"  {symptom}: severity {severity}")
    
    # Test character span extraction for weak supervision
    print("\nCharacter Spans (for weak supervision):")
    spans = extract_character_spans(test_text)
    for start, end, label in spans:
        span_text = test_text[start:end]
        print(f"  {start}-{end}: '{span_text}' -> {label}")
    
    # Run auto-labeling on a sample dataset
    print("\nRunning auto-labeling on sample data...")
    auto_label_sentences('data/ai-medical-chatbot.csv', 'data/auto_labeled.jsonl', max_sentences=150)
    
    # Validate the auto-labeling results
    validate_spans('data/auto_labeled.jsonl', num_examples=3)
