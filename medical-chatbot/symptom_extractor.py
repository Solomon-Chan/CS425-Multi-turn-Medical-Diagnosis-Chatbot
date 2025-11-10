"""
Stage 2: ML-Based Symptom Extractor with Confirmation Flow

This module handles symptom extraction using a fine-tuned SentenceTransformer model.
It extracts symptoms from natural language and returns ranked candidates for confirmation.
"""

from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ExtractedSymptom:
    """Represents a symptom extracted from patient input."""
    symptom: str
    confidence: float
    
    def to_dict(self):
        return {
            'symptom': self.symptom,
            'confidence': self.confidence,
            'percentage': f"{self.confidence * 100:.1f}%"
        }


class MLSymptomExtractor:
    """ML-based symptom extraction using fine-tuned SentenceTransformer."""
    
    def __init__(self, 
                #  model_path: str = 'models/transformer_original/medical_symptom_matcher',
                 model_path: str = 'models\\transformer_updated\\sentencetransformer_package\\sentence_transformer_best',
                 symptoms_path: str = 'data/symptoms.csv',
                 threshold: float = 0.5,
                 top_k: int = 5):
        """
        Initialize the ML symptom extractor.
        
        Args:
            model_path: Path to fine-tuned model
            symptoms_path: Path to canonical symptoms CSV
            threshold: Minimum confidence threshold (0.0-1.0)
            top_k: Maximum symptoms to return per extraction
        """
        print(f"Loading ML symptom extractor from {model_path}...")
        
        # Load model
        self.model = SentenceTransformer(model_path)
        
        # Load canonical symptoms
        symptoms_df = pd.read_csv(symptoms_path)
        self.canonical_symptoms = symptoms_df['symptoms'].tolist()
        
        # Pre-compute embeddings for all symptoms
        print(f"Computing embeddings for {len(self.canonical_symptoms)} symptoms...")
        self.symptom_embeddings = self.model.encode(
            self.canonical_symptoms, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        self.threshold = threshold
        self.top_k = top_k
        
        print(f"✓ Symptom extractor ready with {len(self.canonical_symptoms)} symptoms")
    
    def extract(self, text: str, threshold: Optional[float] = None, 
                top_k: Optional[int] = None) -> List[ExtractedSymptom]:
        """
        Extract symptoms from patient's natural language input.
        
        Args:
            text: Patient's description
            threshold: Override default confidence threshold
            top_k: Override default top_k
            
        Returns:
            List of ExtractedSymptom objects, sorted by confidence
        """
        if not text or not text.strip():
            return []
        
        # Use instance defaults if not overridden
        threshold = threshold if threshold is not None else self.threshold
        top_k = top_k if top_k is not None else self.top_k
        
        # Encode patient input
        input_embedding = self.model.encode(text, convert_to_tensor=True)
        
        # Calculate similarity with all canonical symptoms
        similarities = util.cos_sim(input_embedding, self.symptom_embeddings)[0]
        
        # Filter by threshold and collect results
        results = []
        for idx, score in enumerate(similarities):
            if score >= threshold:
                results.append(ExtractedSymptom(
                    symptom=self.canonical_symptoms[idx],
                    confidence=float(score)
                ))
        
        # Sort by confidence (highest first) and limit to top_k
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:top_k]
    
    def extract_ranked(self, text: str) -> List[ExtractedSymptom]:
        """
        Extract symptoms and return ranked list for confirmation flow.
        Returns top 5 candidates for the confirmation workflow.
        """
        return self.extract(text, threshold=self.threshold, top_k=5)


def test_extractor():
    """Test the symptom extractor with sample inputs."""
    extractor = MLSymptomExtractor()
    
    test_cases = [
        "I have a headache and feel dizzy",
        "My joints are aching",
        "I can't breathe properly",
        "I feel like throwing up"
    ]
    
    print("\n" + "="*70)
    print("Testing ML Symptom Extractor")
    print("="*70)
    
    for text in test_cases:
        print(f"\nPatient says: \"{text}\"")
        print("-" * 70)
        
        results = extractor.extract_ranked(text)
        
        if results:
            print(f"✅ Detected {len(results)} symptom(s):\n")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.symptom}")
                print(f"     Confidence: {result.to_dict()['percentage']}")
        else:
            print("❌ No symptoms detected")
        
        print()


if __name__ == "__main__":
    test_extractor()
