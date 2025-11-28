#!/usr/bin/env python3
"""
Auto-labeling script for medical symptom extraction with character spans.

This script performs weak supervision by automatically labeling sentences from
the medical chatbot dataset with character spans for symptoms. It uses longest-first
substring matching to extract symptoms and their positions in the text.

Usage:
    python scripts/auto_label.py [--input CSV_FILE] [--output JSONL_FILE] [--max-sentences N]

Features:
- Longest-first substring matching to prevent short matches overshadowing long ones
- Character span extraction for precise tokenizer mapping
- Word boundary detection to avoid partial word matches
- Overlap prevention to ensure clean span boundaries
- JSONL output format for easy processing
"""

import argparse
import json
import os
import re
import sys
from typing import List, Dict, Set, Tuple

import pandas as pd


def load_symptom_data(data_dir: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Load symptom data from JSON files.
    
    Args:
        data_dir: Directory containing symptoms.json and synonyms.json
        
    Returns:
        Tuple of (symptom_list, synonym_map)
    """
    # Load canonical symptoms
    symptoms_file = os.path.join(data_dir, 'symptoms.json')
    with open(symptoms_file, 'r', encoding='utf-8') as f:
        symptom_list = json.load(f)
    
    # Load synonym mapping
    synonyms_file = os.path.join(data_dir, 'synonyms.json')
    with open(synonyms_file, 'r', encoding='utf-8') as f:
        synonym_map = json.load(f)
    
    print(f"Loaded {len(symptom_list)} canonical symptoms and {len(synonym_map)} synonym mappings")
    return symptom_list, synonym_map


def canonicalize_symptom(symptom: str) -> str:
    """
    Canonicalize a symptom by converting to lowercase and removing punctuation.
    
    This function normalizes symptom text for consistent matching by:
    - Converting to lowercase
    - Removing punctuation (except spaces and hyphens)
    - Normalizing whitespace
    - Stripping leading/trailing whitespace
    
    Args:
        symptom: Raw symptom string
        
    Returns:
        Canonicalized symptom string
    """
    # Convert to lowercase
    canonical = symptom.lower()
    
    # Remove punctuation except spaces and hyphens
    canonical = re.sub(r'[^\w\s-]', '', canonical)
    
    # Replace multiple spaces with single space
    canonical = re.sub(r'\s+', ' ', canonical)
    
    # Strip leading/trailing whitespace
    canonical = canonical.strip()
    
    return canonical


def extract_character_spans(text: str, symptom_list: List[str], synonym_map: Dict[str, str]) -> List[Tuple[int, int, str]]:
    """
    Extract character spans for symptoms using longest-first substring matching.
    
    This function performs sophisticated symptom extraction by:
    1. Creating a comprehensive dictionary of all phrases to match
    2. Sorting phrases by length (longest first) to prevent short matches overshadowing long ones
    3. Using word boundary detection to ensure complete word matches
    4. Tracking used character positions to avoid overlapping matches
    5. Returning precise character spans for tokenizer compatibility
    
    Args:
        text: Input text to extract spans from
        symptom_list: List of canonical symptoms
        synonym_map: Dictionary mapping synonyms to canonical symptoms
        
    Returns:
        List of (start_char, end_char, canonical_label) tuples
    """
    if not text:
        return []
    
    spans = []
    
    # Create a comprehensive dictionary of all phrases to match
    # Include both canonical symptoms and synonyms
    all_phrases = {}
    
    # Add canonical symptoms
    for symptom in symptom_list:
        canonical_symptom = canonicalize_symptom(symptom)
        all_phrases[canonical_symptom] = symptom
    
    # Add synonyms
    for synonym, canonical_symptom in synonym_map.items():
        canonical_synonym = canonicalize_symptom(synonym)
        all_phrases[canonical_synonym] = canonical_symptom
    
    # Sort phrases by length (longest first) to prevent short matches overshadowing long ones
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
            is_word_boundary = True
            
            # Check character before the match
            if pos > 0 and text[pos-1].isalnum():
                is_word_boundary = False
            
            # Check character after the match
            end_pos = pos + len(phrase)
            if end_pos < len(text) and text[end_pos].isalnum():
                is_word_boundary = False
            
            # Only add if it's a word boundary match and doesn't overlap
            if is_word_boundary:
                phrase_positions = set(range(pos, end_pos))
                
                if not phrase_positions.intersection(used_positions):
                    # No overlap, add this span
                    spans.append((pos, end_pos, canonical_label))
                    used_positions.update(phrase_positions)
            
            # Move to next potential match
            start = pos + 1
    
    # Sort spans by start position
    spans.sort(key=lambda x: x[0])
    
    return spans


def auto_label_sentences(csv_file: str, output_file: str, data_dir: str, max_sentences: int = 200) -> None:
    """
    Auto-label sentences from the medical chatbot dataset with character spans.
    
    This function processes the medical chatbot dataset by:
    1. Loading symptom data from JSON files
    2. Reading the CSV file in chunks to handle large files efficiently
    3. Processing each text column (Description, Patient, Doctor)
    4. Extracting character spans for symptoms in each sentence
    5. Saving results in JSONL format for easy processing
    
    Args:
        csv_file: Path to the input CSV file
        output_file: Path to the output JSONL file
        data_dir: Directory containing symptom data files
        max_sentences: Maximum number of sentences to process
    """
    print(f"Loading symptom data from {data_dir}...")
    symptom_list, synonym_map = load_symptom_data(data_dir)
    
    print(f"Loading dataset from {csv_file}...")
    
    # Read the CSV file in chunks to handle large files
    chunk_size = 1000
    processed_sentences = 0
    labeled_data = []
    
    try:
        # Read CSV in chunks
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            if processed_sentences >= max_sentences:
                break
                
            for _, row in chunk.iterrows():
                if processed_sentences >= max_sentences:
                    break
                
                # Process each column that contains text
                for column in ['Description', 'Patient', 'Doctor']:
                    if column in row and pd.notna(row[column]):
                        text = str(row[column]).strip()
                        if text and len(text) > 10:  # Skip very short texts
                            # Extract character spans
                            spans = extract_character_spans(text, symptom_list, synonym_map)
                            
                            # Create labeled entry
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
    
    # Save to JSONL format
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in labeled_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Successfully saved {len(labeled_data)} labeled sentences to {output_file}")
        
        # Print some statistics
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
        num_examples: Number of examples to show
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
                
                if spans:  # Only show examples with spans
                    print(f"\nExample {examples_shown + 1}:")
                    print(f"Text: {text}")
                    print(f"Spans: {spans}")
                    
                    # Show highlighted text
                    highlighted = text
                    # Sort spans by start position in reverse order to avoid index shifting
                    sorted_spans = sorted(spans, key=lambda x: x[0], reverse=True)
                    
                    for start, end, label in sorted_spans:
                        span_text = text[start:end]
                        highlighted = highlighted[:start] + f"[{span_text}]({label})" + highlighted[end:]
                    
                    print(f"Highlighted: {highlighted}")
                    examples_shown += 1
                    
    except Exception as e:
        print(f"Error validating spans: {e}")


def main():
    """
    Main function to run the auto-labeling script.
    
    This function handles command-line arguments and orchestrates the auto-labeling process.
    It provides a user-friendly interface for running the symptom extraction pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Auto-label medical symptoms with character spans for weak supervision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/auto_label.py
  python scripts/auto_label.py --max-sentences 500
  python scripts/auto_label.py --input data/custom.csv --output data/custom_labeled.jsonl
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        default='data/ai-medical-chatbot.csv',
        help='Input CSV file path (default: data/ai-medical-chatbot.csv)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='data/auto_labeled.jsonl',
        help='Output JSONL file path (default: data/auto_labeled.jsonl)'
    )
    
    parser.add_argument(
        '--data-dir', '-d',
        default='data',
        help='Directory containing symptoms.json and synonyms.json (default: data)'
    )
    
    parser.add_argument(
        '--max-sentences', '-m',
        type=int,
        default=200,
        help='Maximum number of sentences to process (default: 200)'
    )
    
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Validate spans after processing'
    )
    
    parser.add_argument(
        '--examples', '-e',
        type=int,
        default=3,
        help='Number of validation examples to show (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found")
        sys.exit(1)
    
    # Check if required JSON files exist
    symptoms_file = os.path.join(args.data_dir, 'symptoms.json')
    synonyms_file = os.path.join(args.data_dir, 'synonyms.json')
    
    if not os.path.exists(symptoms_file):
        print(f"Error: Symptoms file '{symptoms_file}' not found")
        print("Run 'python scripts/extract_symptom_data.py' first to create the required JSON files")
        sys.exit(1)
    
    if not os.path.exists(synonyms_file):
        print(f"Error: Synonyms file '{synonyms_file}' not found")
        print("Run 'python scripts/extract_symptom_data.py' first to create the required JSON files")
        sys.exit(1)
    
    print("Auto-labeling Medical Symptoms with Character Spans")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Data directory: {args.data_dir}")
    print(f"Max sentences: {args.max_sentences}")
    print("=" * 60)
    
    try:
        # Run auto-labeling
        auto_label_sentences(args.input, args.output, args.data_dir, args.max_sentences)
        
        # Validate if requested
        if args.validate:
            validate_spans(args.output, args.examples)
        
        print("\nAuto-labeling completed successfully!")
        
    except Exception as e:
        print(f"Error during auto-labeling: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
