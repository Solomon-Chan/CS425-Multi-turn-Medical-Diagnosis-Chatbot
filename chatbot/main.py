#!/usr/bin/env python3
"""
Main chatbot application with fine-tuned BioBERT NER for symptom extraction.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import config
from scripts.infer_ner import load_model_and_tokenizer, predict
from chatbot.symptom_extractor import load_symptom_data, get_symptom_severity, canonicalize_symptom


def extract_symptom_phrases(token_label_pairs):
    """
    Extract complete symptom phrases from token-label pairs.
    Groups consecutive B-SYM/I-SYM tokens into full phrases.
    """
    symptoms = []
    current_symptom = []
    
    for token, label in token_label_pairs:
        if label == "B-SYM":
            # Save previous symptom if exists
            if current_symptom:
                symptoms.append(" ".join(current_symptom))
            current_symptom = [token]
        elif label == "I-SYM":
            # Continue current symptom
            if current_symptom:
                current_symptom.append(token)
        else:  # "O"
            # End current symptom
            if current_symptom:
                symptoms.append(" ".join(current_symptom))
                current_symptom = []
    
    # Don't forget the last symptom
    if current_symptom:
        symptoms.append(" ".join(current_symptom))
    
    return symptoms


def normalize_token(token):
    """Normalize BERT subword tokens (remove ## prefix)."""
    return token.replace("##", "")


def extract_symptoms_ner(text, tokenizer, model):
    """
    Extract symptoms from text using fine-tuned NER model.
    Returns list of symptom phrases.
    """
    # Get token-level predictions
    token_label_pairs = predict(text, tokenizer, model)
    
    # Normalize tokens (handle subword tokens like ##ing -> ing)
    normalized_pairs = [(normalize_token(token), label) for token, label in token_label_pairs]
    
    # Extract symptom phrases
    symptoms = extract_symptom_phrases(normalized_pairs)
    
    # Clean up symptoms (remove empty, deduplicate)
    symptoms = [s.strip() for s in symptoms if s.strip()]
    symptoms = list(dict.fromkeys(symptoms))  # Preserve order while deduplicating
    
    return symptoms


class MedicalChatbot:
    """
    Medical diagnosis chatbot with symptom extraction using fine-tuned BioBERT NER.
    """
    
    def __init__(self, model_path=None):
        """Initialize chatbot with NER model."""
        config.validate_paths()
        self.model_path = model_path or config.MODEL_PATH
        self.tokenizer, self.model = load_model_and_tokenizer(self.model_path)
        self.model.eval()
        
        # Load symptom data for validation and severity
        self.symptoms, self.synonyms = load_symptom_data(str(config.DATA_DIR), verbose=True)
        print(f"✓ Loaded model from {self.model_path}")
        print(f"✓ Loaded {len(self.symptoms)} canonical symptoms from data directory")
    
    def extract_symptoms(self, user_input):
        """
        Extract symptoms from user input using NER model.
        
        Args:
            user_input: Patient's text input
            
        Returns:
            List of extracted symptom phrases
        """
        if not user_input or not user_input.strip():
            return []
        
        # Step 1: extract surface symptom phrases from NER
        surface_symptoms = extract_symptoms_ner(user_input, self.tokenizer, self.model)

        # Step 2: map each extracted phrase to a canonical symptom (if possible)
        canonical_matches = []
        seen = set()

        # Precompute normalized canonical map for fast lookup
        norm_to_canonical = {canonicalize_symptom(s): s for s in self.symptoms}

        # Normalize synonyms map keys as well
        norm_syn_map = {canonicalize_symptom(k): v for k, v in self.synonyms.items()}

        for phrase in surface_symptoms:
            norm = canonicalize_symptom(phrase)
            canon = None
            # Exact canonical match
            if norm in norm_to_canonical:
                canon = norm_to_canonical[norm]
            # Synonym match
            elif norm in norm_syn_map:
                canon = norm_syn_map[norm]
            else:
                # As a fallback, try substring matching against canonical symptoms
                for k_norm, canon_cand in norm_to_canonical.items():
                    if k_norm in norm or norm in k_norm:
                        canon = canon_cand
                        break

            if canon and canon not in seen:
                canonical_matches.append(canon)
                seen.add(canon)

        return canonical_matches
    
    def analyze_symptoms(self, symptoms):
        """
        Analyze extracted symptoms and return severity information.
        
        Args:
            symptoms: List of symptom phrases
            
        Returns:
            Dict with symptom analysis including severity scores
        """
        analysis = {
            "symptoms": symptoms,
            "count": len(symptoms),
            "severities": {}
        }
        
        # Get severity for each symptom
        total_severity = 0
        for symptom in symptoms:
            # Try exact match first
            severity = get_symptom_severity(symptom)
            if severity == 0:
                # Try canonicalizing
                from chatbot.symptom_extractor import canonicalize_symptom
                canon_symptom = canonicalize_symptom(symptom)
                # Check if it matches any canonical symptom
                for canonical in self.symptoms:
                    if canonicalize_symptom(canonical) == canon_symptom:
                        severity = get_symptom_severity(canonical)
                        break
            
            analysis["severities"][symptom] = severity
            total_severity += severity
        
        analysis["total_severity"] = total_severity
        analysis["average_severity"] = total_severity / len(symptoms) if symptoms else 0
        
        return analysis
    
    def chat(self, user_input):
        """
        Process user input and extract symptoms.
        
        Args:
            user_input: Patient's text input
            
        Returns:
            Dict with extracted symptoms and analysis
        """
        symptoms = self.extract_symptoms(user_input)
        
        if not symptoms:
            return {
                "symptoms": [],
                "message": "No symptoms detected in your input. Please describe your symptoms in more detail."
            }
        
        analysis = self.analyze_symptoms(symptoms)
        
        return {
            "symptoms": symptoms,
            "analysis": analysis,
            "message": f"I detected {len(symptoms)} symptom(s): {', '.join(symptoms)}"
        }


def main():
    """Interactive chatbot CLI."""
    print("=" * 60)
    print("Medical Diagnosis Chatbot with BioBERT NER")
    print("=" * 60)
    print("Type 'quit' or 'exit' to stop\n")
    
    # Initialize chatbot
    try:
        chatbot = MedicalChatbot()
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        return
    
    # Interactive loop
    while True:
        try:
            user_input = input("\nPatient: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the Medical Diagnosis Chatbot!")
                break
            
            # Extract symptoms
            result = chatbot.chat(user_input)
            
            # Display results
            print(f"\nChatbot: {result['message']}")
            
            if result.get('symptoms'):
                print("\nExtracted Symptoms:")
                for i, symptom in enumerate(result['symptoms'], 1):
                    severity = result['analysis']['severities'].get(symptom, 0)
                    severity_indicator = f" (severity: {severity})" if severity > 0 else ""
                    print(f"  {i}. {symptom}{severity_indicator}")
                
                if result['analysis']['total_severity'] > 0:
                    print(f"\nTotal severity score: {result['analysis']['total_severity']}")
        
        except KeyboardInterrupt:
            print("\n\nThank you for using the Medical Diagnosis Chatbot!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

