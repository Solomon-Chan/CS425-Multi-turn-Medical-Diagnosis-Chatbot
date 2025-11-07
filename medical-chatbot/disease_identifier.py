"""
Stage 3: Two-Tier Disease Identification with BioGPT

This module handles disease diagnosis using:
1. Common illness fast matching (Tier 1)
2. BioGPT + MedQuAD extraction (Tier 2)
3. Unique symptom identification for disease disambiguation
"""

from transformers import BioGptForCausalLM, BioGptTokenizer
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import json
import re
from typing import List, Dict, Optional, Tuple
from collections import Counter


class DiseaseIdentifier:
    """Two-tier disease identification system with unique symptom detection."""
    
    def __init__(self,
                 common_illnesses_path: str = 'data/common_illnesses.json',
                 medquad_path: str = 'data/medquad.csv',
                 embeddings_path: str = 'models/corpus_embeddings.pt',
                 embedder_model: str = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'):
        """
        Initialize the disease identifier.
        
        Args:
            common_illnesses_path: Path to common illnesses JSON
            medquad_path: Path to MedQuAD CSV database
            embeddings_path: Path to pre-computed corpus embeddings
            embedder_model: BioBERT model for embeddings
        """
        print("Initializing disease identifier...")
        
        # Load common illnesses for Tier 1
        print("Loading common illnesses...")
        with open(common_illnesses_path, 'r') as f:
            self.common_conditions = json.load(f)
        print(f"✓ Loaded {len(self.common_conditions)} common conditions")
        
        # Load MedQuAD database
        print("Loading MedQuAD database...")
        self.medquad_df = pd.read_csv(medquad_path)
        
        # Clean the data - remove rows with NaN in focus_area
        original_len = len(self.medquad_df)
        self.medquad_df = self.medquad_df.dropna(subset=['focus_area'])
        self.medquad_df = self.medquad_df[self.medquad_df['focus_area'].str.strip() != '']
        
        if len(self.medquad_df) < original_len:
            print(f"  Cleaned {original_len - len(self.medquad_df)} invalid entries")
        
        self.medquad_df['text'] = (
            self.medquad_df['question'] + " " + self.medquad_df['answer']
        ).str.strip()
        print(f"✓ Loaded {len(self.medquad_df)} medical Q&A pairs")
        
        # Load embedder for semantic search
        print("Loading BioBERT embedder...")
        self.embedder = SentenceTransformer(embedder_model)
        
        # Load pre-computed embeddings
        print("Loading pre-computed embeddings...")
        data = torch.load(embeddings_path)
        self.corpus_embeddings = data['embeddings']
        
        # Validate embeddings match dataframe
        if len(self.corpus_embeddings) != len(self.medquad_df):
            print(f"⚠️  WARNING: Embeddings count ({len(self.corpus_embeddings)}) doesn't match MedQuAD rows ({len(self.medquad_df)})")
            print(f"  This may happen if embeddings were created before data cleaning.")
            print(f"  Diagnosis will still work but may be less accurate.")
            print(f"  Consider regenerating embeddings with the cleaned dataset.")
        
        print(f"✓ Loaded {len(self.corpus_embeddings)} embeddings")
        
        # Load BioGPT for diagnosis
        print("Loading BioGPT model...")
        self.biogpt_model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
        self.biogpt_tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.biogpt_model = self.biogpt_model.to(self.device)
        print(f"✓ BioGPT loaded on {self.device}")
        
        print("✓ Disease identifier ready")
    
    def regenerate_embeddings(self, save_path: str = 'models/corpus_embeddings_cleaned.pt'):
        """
        Regenerate embeddings for the cleaned MedQuAD dataset.
        Call this if you get warnings about embeddings mismatch.
        
        Args:
            save_path: Where to save the new embeddings
        """
        print(f"\n🔄 Regenerating embeddings for {len(self.medquad_df)} entries...")
        print("This may take several minutes...")
        
        corpus_texts = self.medquad_df['text'].tolist()
        
        # Encode in batches
        self.corpus_embeddings = self.embedder.encode(
            corpus_texts,
            batch_size=64,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        # Save
        torch.save({
            'texts': corpus_texts,
            'embeddings': self.corpus_embeddings
        }, save_path)
        
        print(f"✓ New embeddings saved to {save_path}")
        print(f"  Update embeddings_path parameter to use: '{save_path}'")
    
    def _normalize_symptom(self, symptom: str) -> str:
        """Normalize symptom for matching."""
        return symptom.lower().strip()
    
    def check_common_conditions(self, symptoms: List[str]) -> Tuple[str, Optional[Dict]]:
        """
        Tier 1: Fast check against common illnesses.
        
        Returns:
            (strategy, match_info) where strategy is:
            - 'common_illness': High confidence match
            - 'hybrid_search': Partial match
            - 'full_medquad': No match, use Tier 2
        """
        normalized_symptoms = [self._normalize_symptom(s) for s in symptoms]
        normalized_set = set(normalized_symptoms)
        
        matches = {}
        
        for condition_key, data in self.common_conditions.items():
            # Normalize condition symptoms
            primary_symptoms = [self._normalize_symptom(s) 
                              for s in data['symptoms']['primary']]
            secondary_symptoms = [self._normalize_symptom(s) 
                                for s in data['symptoms']['secondary']]
            
            # Calculate overlaps
            primary_matches = normalized_set & set(primary_symptoms)
            secondary_matches = normalized_set & set(secondary_symptoms)
            
            # Score: primary worth 2 points, secondary worth 1
            total_score = len(primary_matches) * 2 + len(secondary_matches)
            
            # Check minimum threshold
            if len(primary_matches) >= data['min_match'] or \
               total_score >= data['min_match'] * 2:
                matches[condition_key] = {
                    'full_name': data['full_name'],
                    'score': total_score,
                    'data': data
                }
        
        if not matches:
            return 'full_medquad', None
        
        # Get best match
        best_condition = max(matches.items(), key=lambda x: x[1]['score'])
        best_score = best_condition[1]['score']
        
        # Decision thresholds
        HIGH_CONFIDENCE = 4
        LOW_CONFIDENCE = 2
        
        if best_score >= HIGH_CONFIDENCE:
            return 'common_illness', best_condition
        elif best_score >= LOW_CONFIDENCE:
            return 'hybrid_search', best_condition
        else:
            return 'full_medquad', None
    
    def diagnose_with_biogpt(self, symptoms: List[str]) -> List[str]:
        """
        Use BioGPT to diagnose disease from symptoms.
        
        Returns:
            List of disease candidates
        """
        symptom_text = ', '.join(symptoms)
        prompt = f"A patient presents with {symptom_text}. The most likely diagnosis is"
        
        try:
            inputs = self.biogpt_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.biogpt_model.generate(
                inputs.input_ids,
                max_length=100,
                num_return_sequences=3,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
            
            # Parse disease names from outputs
            candidates = []
            for output in outputs:
                text = self.biogpt_tokenizer.decode(output, skip_special_tokens=True)
                # Extract disease name (after "diagnosis is")
                if "diagnosis is" in text.lower():
                    disease_part = text.lower().split("diagnosis is")[-1]
                    disease = disease_part.split('.')[0].split(',')[0].strip()
                    if disease and len(disease) < 100:
                        candidates.append(disease)
            
            return candidates if candidates else None
        
        except Exception as e:
            print(f"BioGPT diagnosis error: {e}")
            return None
    
    def find_closest_medquad_disease(self, biogpt_disease: str) -> Optional[str]:
        """Match BioGPT diagnosis to MedQuAD database."""
        if not biogpt_disease:
            return None
        
        biogpt_lower = biogpt_disease.lower()
        
        # Get all unique diseases and filter out NaN/None values
        all_diseases = [d for d in self.medquad_df['focus_area'].unique().tolist() 
                       if isinstance(d, str)]
        
        # Try exact match
        if biogpt_disease in all_diseases:
            return biogpt_disease
        
        # Try case-insensitive
        for disease in all_diseases:
            if disease.lower() == biogpt_lower:
                return disease
        
        # Try partial match
        for disease in all_diseases:
            if biogpt_lower in disease.lower() or disease.lower() in biogpt_lower:
                return disease
        
        # Last resort: semantic search
        query_emb = self.embedder.encode([biogpt_disease], convert_to_tensor=True)
        disease_texts = [f"{d}" for d in all_diseases]
        disease_embs = self.embedder.encode(disease_texts, convert_to_tensor=True)
        
        hits = util.semantic_search(query_emb, disease_embs, top_k=1)[0]
        
        if hits:
            return all_diseases[hits[0]['corpus_id']]
        
        return None
    
    def predict_disease_medquad(self, symptoms: List[str], top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Semantic search through MedQuAD for disease prediction.
        
        Returns:
            List of (disease, score) tuples
        """
        try:
            query = "Patient presents with " + ", ".join(symptoms)
            q_emb = self.embedder.encode([query], convert_to_tensor=True)
            
            # Ensure top_k doesn't exceed corpus size
            actual_top_k = min(top_k, len(self.corpus_embeddings))
            
            hits = util.semantic_search(q_emb, self.corpus_embeddings, top_k=actual_top_k)[0]
            
            # Aggregate by disease
            disease_scores = Counter()
            for h in hits:
                idx = h['corpus_id']
                
                # Check if index is within bounds
                if idx < len(self.medquad_df):
                    disease = self.medquad_df.iloc[idx]['focus_area']
                    
                    # Only add if disease is valid (not NaN)
                    if isinstance(disease, str) and disease.strip():
                        score = h['score']
                        disease_scores[disease] += score
            
            return disease_scores.most_common(10)
        
        except Exception as e:
            print(f"Error in predict_disease_medquad: {e}")
            return []
    
    def retrieve_facts_for_disease(self, disease: str, top_k: int = 12) -> List[Dict]:
        """Retrieve Q&A facts for a specific disease."""
        mask = (self.medquad_df['focus_area'] == disease)
        subset = self.medquad_df[mask]
        
        if subset.empty:
            return []
        
        texts = subset['text'].tolist()
        embs = self.embedder.encode(texts, convert_to_tensor=True)
        
        # Use disease name as query
        q_emb = self.embedder.encode([disease], convert_to_tensor=True)
        hits = util.semantic_search(q_emb, embs, top_k=min(top_k, len(texts)))[0]
        
        picked = []
        sub_q = subset['question'].tolist()
        sub_a = subset['answer'].tolist()
        
        for h in hits:
            i = h['corpus_id']
            picked.append({
                'question': sub_q[i],
                'answer': sub_a[i]
            })
        
        return picked
    
    def extract_description(self, disease: str, facts: List[Dict], max_sentences: int = 5) -> str:
        """Extract disease description from Q&A facts."""
        if not facts:
            return f"No detailed information available for {disease}."
        
        description_keywords = ['what is', 'what are', 'define', 'description', 'overview']
        relevant_answers = []
        
        # Look for definition questions
        for fact in facts[:10]:
            q_lower = fact['question'].lower()
            if any(keyword in q_lower for keyword in description_keywords):
                answer = fact['answer'].strip()
                if len(answer) > 50:
                    relevant_answers.append(answer)
                    if len(relevant_answers) >= 2:
                        break
        
        # Fallback to first substantial answer
        if not relevant_answers:
            for fact in facts[:3]:
                answer = fact['answer'].strip()
                if len(answer) > 50:
                    relevant_answers.append(answer)
                    break
        
        if not relevant_answers:
            return f"Limited information available for {disease}."
        
        # Combine and limit
        description = ' '.join(relevant_answers[:2])
        sentences = [s.strip() + '.' for s in description.split('.') if s.strip()]
        
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        
        return ' '.join(sentences)
    
    def extract_recommendations(self, disease: str, facts: List[Dict]) -> str:
        """Extract recommendations from Q&A facts."""
        if not facts:
            return "Consult with a healthcare provider for personalized recommendations."
        
        recommendation_keywords = [
            'prevent', 'avoid', 'should not', 'do not', 'management',
            'lifestyle', 'precaution', 'risk', 'complication', 'warning'
        ]
        
        relevant_answers = []
        
        for fact in facts:
            q_lower = fact['question'].lower()
            if any(keyword in q_lower for keyword in recommendation_keywords):
                answer = fact['answer'].strip()
                if len(answer) > 30:
                    relevant_answers.append(answer)
                    if len(relevant_answers) >= 3:
                        break
        
        # Fallback
        if not relevant_answers:
            fallback_keywords = ['treatment', 'outlook', 'prognosis', 'care']
            for fact in facts:
                q_lower = fact['question'].lower()
                if any(keyword in q_lower for keyword in fallback_keywords):
                    answer = fact['answer'].strip()
                    if len(answer) > 30:
                        relevant_answers.append(answer)
                        if len(relevant_answers) >= 2:
                            break
        
        if not relevant_answers:
            return "Consult with a healthcare provider for specific guidance."
        
        # Combine
        recommendations = ' '.join(relevant_answers[:3])
        sentences = [s.strip() + '.' for s in recommendations.split('.') if s.strip()]
        
        if len(sentences) > 6:
            sentences = sentences[:6]
        
        return ' '.join(sentences)
    
    def get_unique_symptoms_for_diseases(self, diseases: List[str], 
                                        all_symptoms: List[str]) -> Dict[str, List[str]]:
        """
        Use BioGPT to identify unique/distinguishing symptoms for candidate diseases.
        
        Args:
            diseases: Top 3 candidate diseases
            all_symptoms: Symptoms already reported by patient
            
        Returns:
            Dict mapping disease to list of unique symptoms to ask about
        """
        symptom_text = ', '.join(all_symptoms)
        disease_text = ', '.join(diseases)
        
        prompt = (f"For a patient with {symptom_text}, the differential diagnosis includes "
                 f"{disease_text}. What are the key distinguishing symptoms for each condition?")
        
        try:
            inputs = self.biogpt_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.biogpt_model.generate(
                inputs.input_ids,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7
            )
            
            text = self.biogpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse unique symptoms (simplified - you may want more sophisticated parsing)
            unique_symptoms = {}
            for disease in diseases:
                unique_symptoms[disease] = []
                # Look for disease name in output and extract nearby symptoms
                # This is a placeholder - implement more robust parsing
                
            return unique_symptoms
        
        except Exception as e:
            print(f"Error getting unique symptoms: {e}")
            return {disease: [] for disease in diseases}
    
    def diagnose(self, symptoms: List[str], verbose: bool = False) -> Dict:
        """
        Complete diagnosis pipeline.
        
        Returns:
            {
                'disease': str,
                'confidence': 'high' | 'medium' | 'low',
                'strategy': str,
                'description': str,
                'recommendation': str,
                'top_candidates': List[str]  # Top 3 for disambiguation
            }
        """
        if verbose:
            print(f"\n🔍 Diagnosing from symptoms: {symptoms}")
        
        # Tier 1: Check common illnesses
        strategy, common_match = self.check_common_conditions(symptoms)
        
        if strategy == 'common_illness':
            # High confidence common illness
            condition_data = common_match[1]['data']
            
            if verbose:
                print(f"✓ Common illness detected: {common_match[1]['full_name']}")
            
            return {
                'disease': common_match[1]['full_name'],
                'confidence': 'high',
                'strategy': 'common_illness',
                'description': condition_data.get('description', ''),
                'recommendation': condition_data.get('recommendation', ''),
                'top_candidates': [common_match[1]['full_name']]
            }
        
        # Tier 2: Use BioGPT + MedQuAD
        if verbose:
            print("  Using BioGPT for diagnosis...")
        
        try:
            biogpt_candidates = self.diagnose_with_biogpt(symptoms)
        except Exception as e:
            if verbose:
                print(f"  BioGPT error: {e}")
            biogpt_candidates = None
        
        disease = None
        
        if biogpt_candidates:
            # Try to match to MedQuAD
            for candidate in biogpt_candidates:
                try:
                    matched = self.find_closest_medquad_disease(candidate)
                    if matched:
                        disease = matched
                        if verbose:
                            print(f"  BioGPT diagnosis: '{candidate}' → '{disease}'")
                        break
                except Exception as e:
                    if verbose:
                        print(f"  Error matching '{candidate}': {e}")
                    continue
            
            if not disease and biogpt_candidates:
                disease = biogpt_candidates[0]
        
        # Fallback to semantic search if BioGPT failed
        if not disease:
            if verbose:
                print("  BioGPT failed, using semantic search...")
            try:
                ranked = self.predict_disease_medquad(symptoms)
                disease = ranked[0][0] if ranked else None
            except Exception as e:
                if verbose:
                    print(f"  Semantic search error: {e}")
                disease = None
        
        if not disease:
            return {
                'disease': None,
                'confidence': 'low',
                'strategy': 'failed',
                'description': 'Unable to determine diagnosis from the provided symptoms.',
                'recommendation': 'Please consult a healthcare provider for proper evaluation.',
                'top_candidates': []
            }
        
        # Extract information from MedQuAD
        try:
            facts = self.retrieve_facts_for_disease(disease, top_k=12)
            description = self.extract_description(disease, facts)
            recommendation = self.extract_recommendations(disease, facts)
        except Exception as e:
            if verbose:
                print(f"  Error extracting info: {e}")
            description = f"Information about {disease} is available, but we encountered an error retrieving details."
            recommendation = "Please consult a healthcare provider for more information about this condition."
        
        # Get top 3 candidates for disambiguation
        try:
            ranked = self.predict_disease_medquad(symptoms, top_k=50)
            top_candidates = [d for d, _ in ranked[:3]]
        except Exception as e:
            if verbose:
                print(f"  Error getting candidates: {e}")
            top_candidates = [disease]
        
        return {
            'disease': disease,
            'confidence': 'medium',
            'strategy': 'biogpt' if biogpt_candidates else 'semantic_search',
            'description': description,
            'recommendation': recommendation,
            'top_candidates': top_candidates
        }


if __name__ == "__main__":
    # Test the disease identifier
    identifier = DiseaseIdentifier()
    
    test_symptoms = ['headache', 'dizziness', 'nausea']
    result = identifier.diagnose(test_symptoms, verbose=True)
    
    print("\n" + "="*70)
    print("Diagnosis Result:")
    print("="*70)
    print(f"Disease: {result['disease']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Strategy: {result['strategy']}")
    print(f"\nDescription:\n{result['description']}")
    print(f"\nRecommendation:\n{result['recommendation']}")
