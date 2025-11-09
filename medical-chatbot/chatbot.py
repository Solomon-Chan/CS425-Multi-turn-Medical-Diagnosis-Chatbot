"""
Stage 1: Medical Chatbot with Confirmation Flow

Main chatbot orchestrating the symptom extraction and disease identification process.
Features a confirmation-based conversation flow to ensure accuracy.
Requires 3 confirmed symptoms before providing a diagnosis using BioBERT.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import spacy
import random


class Phase(Enum):
    """Conversation phases."""
    COLLECT = "collect"
    CONFIRM_SYMPTOM = "confirm_symptom"
    CONFIRM_SECONDARY = "confirm_secondary"
    DIAGNOSE = "diagnose"
    EXPLAIN = "explain"


@dataclass
class ChatState:
    """Manages conversation state."""
    confirmed_symptoms: List[str] = field(default_factory=list)
    pending_confirmation: Optional[str] = None
    pending_secondary: Optional[str] = None
    candidates_queue: List[str] = field(default_factory=list)
    phase: Phase = Phase.COLLECT
    turn: int = 0
    k_threshold: int = 3  # Collect 3 confirmed symptoms
    last_extraction: List = field(default_factory=list)
    diagnosis_result: Optional[Dict] = None
    
    def add_confirmed_symptom(self, symptom: str):
        """Add a confirmed symptom."""
        if symptom not in self.confirmed_symptoms:
            self.confirmed_symptoms.append(symptom)
    
    def symptom_count(self) -> int:
        """Get count of confirmed symptoms."""
        return len(self.confirmed_symptoms)
    
    def ready_for_diagnosis(self) -> bool:
        """Check if we have enough symptoms for diagnosis."""
        return self.symptom_count() >= self.k_threshold
    
    def reset(self):
        """Reset the conversation state."""
        self.confirmed_symptoms = []
        self.pending_confirmation = None
        self.pending_secondary = None
        self.candidates_queue = []
        self.phase = Phase.COLLECT
        self.turn = 0
        self.last_extraction = []
        self.diagnosis_result = None


class MedicalChatbot:
    """Main chatbot orchestrating the confirmation flow."""
    
    def __init__(self, symptom_extractor, disease_identifier, intent_model=None):
        """
        Initialize chatbot with ML components.
        
        Args:
            symptom_extractor: MLSymptomExtractor instance
            disease_identifier: DiseaseIdentifier instance
        """
        self.extractor = symptom_extractor
        self.identifier = disease_identifier
        self.intent_model = intent_model
        self.state = ChatState()
    
    def welcome(self) -> str:
        """Welcome message."""
        return (
            "Hello! I'm a medical symptom assessment assistant. "
            "I'll help you understand your symptoms and provide guidance.\n\n"
            "⚠️ Important: This is for informational purposes only and not a substitute "
            "for professional medical advice.\n\n"
            "📋 Please describe ONE symptom at a time for the most accurate assessment.\n\n"
            "What symptoms are you experiencing?"
        )
    
    def process_message(self, text: str) -> str:
        """
        Process user message and return response.
        
        Args:
            text: User's message
            
        Returns:
            Bot's response
        """
        self.state.turn += 1
        text = text.strip()
        
        if not text:
            return "Could you describe your symptoms?"
        
        # Handle commands
        lower_text = text.lower()
        if lower_text in ['restart', 'start over']:
            self.state.reset()
            return "Okay, let's start over. What symptoms are you experiencing?"
        
        if lower_text == 'list':
            if self.state.confirmed_symptoms:
                return f"Confirmed symptoms: {', '.join(self.state.confirmed_symptoms)}"
            return "No symptoms confirmed yet."
        
        if lower_text == 'help':
            return (
                "Commands:\n"
                "• 'list' - Show confirmed symptoms\n"
                "• 'restart' - Start over\n"
                "• 'help' - Show this message\n\n"
                "Just describe your symptoms naturally, and I'll guide you through the process."
            )
        
        if self.state.phase in [Phase.CONFIRM_SYMPTOM, Phase.CONFIRM_SECONDARY, Phase.DIAGNOSE]:
            # Skip small talk detection in these phases
            pass
        else:
            intent_doc = self.intent_model(text)
            intent = max(intent_doc.cats, key=intent_doc.cats.get)
            if intent == "small_talk" and intent_doc.cats[intent] > 0.7:
                responses = [
                    "Let's focus on your health. Could you describe any symptoms you're feeling?",
                    "I’m here to help with medical concerns — are you experiencing any symptoms?",
                    "Sure thing! But first, can you tell me what symptoms you have?"
                ]
                return random.choice(responses)
        
        # Route based on phase
        if self.state.phase == Phase.COLLECT:
            return self._handle_collect_phase(text)
        
        elif self.state.phase == Phase.CONFIRM_SYMPTOM:
            return self._handle_confirm_symptom(text)
        
        elif self.state.phase == Phase.CONFIRM_SECONDARY:
            return self._handle_confirm_secondary(text)
        
        elif self.state.phase == Phase.DIAGNOSE:
            return self._handle_diagnose_phase(text)
        
        elif self.state.phase == Phase.EXPLAIN:
            return "Thank you. If you have more symptoms, type 'restart' to begin again."
        
        return "I'm not sure how to help with that. Could you describe your symptoms?"
    
    def _handle_collect_phase(self, text: str) -> str:
        """Handle initial symptom collection."""
        # Extract symptoms from user input
        extracted = self.extractor.extract_ranked(text)
        
        if not extracted:
            return "I didn't detect any clear symptoms. Could you describe what you're feeling?"
        
        # Store extraction for confirmation flow
        self.state.last_extraction = extracted
        
        # Take top symptom for confirmation
        top_symptom = extracted[0]
        self.state.pending_confirmation = top_symptom.symptom
        
        # Store remaining for secondary confirmations
        if len(extracted) > 1:
            self.state.candidates_queue = [e.symptom for e in extracted[1:]]
        else:
            self.state.candidates_queue = []
        
        # Move to confirmation phase
        self.state.phase = Phase.CONFIRM_SYMPTOM
        
        return f"I'm hearing that you have {top_symptom.symptom}. Is that right?"
    
    def _handle_confirm_symptom(self, text: str) -> str:
        """Handle confirmation of primary symptom."""
        response = text.lower().strip()
        
        if response in ['yes', 'y', 'yeah', 'correct', 'right', 'yep']:
            # Confirmed! Add it
            symptom = self.state.pending_confirmation
            self.state.add_confirmed_symptom(symptom)
            self.state.pending_confirmation = None
            
            # Check if we have enough symptoms
            if self.state.ready_for_diagnosis():
                return self._move_to_diagnosis()
            
            # Ask about secondary symptom if available
            if self.state.candidates_queue:
                secondary = self.state.candidates_queue.pop(0)
                self.state.pending_secondary = secondary
                self.state.phase = Phase.CONFIRM_SECONDARY
                
                return (f"Noted: {symptom}. "
                       f"Do you also have {secondary}?")
            else:
                # No more candidates, ask for more symptoms
                self.state.phase = Phase.COLLECT
                return (f"Noted: {symptom}. "
                       f"What other symptoms do you have?")
        
        elif response in ['no', 'n', 'nope', 'incorrect', 'wrong']:
            # Rejected, move on
            self.state.pending_confirmation = None
            self.state.phase = Phase.COLLECT
            
            return "I see. What other symptoms are you experiencing?"
        
        else:
            # Unclear response
            return f"Please answer 'yes' or 'no' - do you have {self.state.pending_confirmation}?"
    
    def _handle_confirm_secondary(self, text: str) -> str:
        """Handle confirmation of secondary symptom."""
        response = text.lower().strip()
        
        if response in ['yes', 'y', 'yeah', 'correct', 'right', 'yep']:
            # Confirmed secondary symptom
            symptom = self.state.pending_secondary
            self.state.add_confirmed_symptom(symptom)
            self.state.pending_secondary = None
            
            # Check if we have enough symptoms
            if self.state.ready_for_diagnosis():
                return self._move_to_diagnosis()
            
            # Continue asking about remaining candidates, but only for #2
            # After that, ask open-ended
            if len(self.state.confirmed_symptoms) == 1 and self.state.candidates_queue:
                # We've confirmed 1 symptom, ask about one more from queue
                secondary = self.state.candidates_queue.pop(0)
                self.state.pending_secondary = secondary
                
                return (f"Noted: {symptom}. "
                       f"Do you also have {secondary}?")
            else:
                # Move to open-ended collection
                self.state.phase = Phase.COLLECT
                return (f"Noted: {symptom}. "
                       f"What other symptoms do you have?")
        
        elif response in ['no', 'n', 'nope', 'incorrect', 'wrong']:
            # Rejected secondary
            self.state.pending_secondary = None
            
            # Check if we have enough symptoms already
            if self.state.ready_for_diagnosis():
                return self._move_to_diagnosis()
            
            # Ask for more symptoms
            self.state.phase = Phase.COLLECT
            return "What other symptoms do you have?"
        
        else:
            return f"Please answer 'yes' or 'no' - do you have {self.state.pending_secondary}?"
    
    def _move_to_diagnosis(self) -> str:
        """Transition to diagnosis phase."""
        self.state.phase = Phase.DIAGNOSE
        
        symptoms_list = ', '.join(self.state.confirmed_symptoms)
        
        return (f"Thank you. I've collected {len(self.state.confirmed_symptoms)} symptoms: "
               f"{symptoms_list}.\n\n"
               f"Would you like me to provide a diagnosis and recommendations? (yes/no)")
    
    def _handle_diagnose_phase(self, text: str) -> str:
        """Handle diagnosis confirmation and execution."""
        response = text.lower().strip()
        
        if response in ['yes', 'y', 'yeah', 'sure', 'ok', 'okay']:
            # Perform diagnosis
            print("🔍 Analyzing symptoms...")
            result = self.identifier.diagnose(self.state.confirmed_symptoms, verbose=True)
            
            self.state.diagnosis_result = result
            
            if not result['disease']:
                self.state.phase = Phase.EXPLAIN
                return (
                    "I'm having trouble determining a specific condition from these symptoms. "
                    "I recommend consulting with a healthcare provider for proper evaluation."
                )
            
            # Format response - go straight to final diagnosis
            confidence_emoji = {
                'high': '✅',
                'medium': '⚠️',
                'low': '❓'
            }
            
            emoji = confidence_emoji.get(result['confidence'], '❓')
            
            response = f"{emoji} **Possible Condition: {result['disease']}**\n\n"
            response += f"**What is it?**\n{result['description']}\n\n"
            response += f"**Recommendations:**\n{result['recommendation']}\n\n"
            response += (
                "**Important:** This is for informational purposes only and not a medical diagnosis. "
                "Please consult a healthcare provider for proper evaluation and treatment."
            )
            
            self.state.phase = Phase.EXPLAIN
            
            return response
        
        elif response in ['no', 'n', 'nope']:
            self.state.phase = Phase.COLLECT
            return "No problem. Would you like to add more symptoms? If so, please describe them."
        
        else:
            return "Please answer 'yes' or 'no' - would you like a diagnosis?"
    
    def run_cli(self):
        """Run chatbot in CLI mode for testing."""
        print(self.welcome())
        print("\n" + "="*70 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye! Take care of yourself.")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Bot: Goodbye! Take care of yourself.")
                break
            
            response = self.process_message(user_input)
            print(f"\nBot: {response}\n")


def create_chatbot():
    """Factory function to create a fully configured chatbot."""
    from symptom_extractor import MLSymptomExtractor
    from disease_identifier import DiseaseIdentifier
    
    print("Initializing medical chatbot...")
    print("="*70)
    
    # Initialize components
    extractor = MLSymptomExtractor()
    print()
    identifier = DiseaseIdentifier()
    print()

    print("Loading intent classifier...")
    intent_model = spacy.load("models/intent_model_e3_v3")
    print("✓ Intent model loaded\n")
    
    # Create chatbot
    chatbot = MedicalChatbot(extractor, identifier, intent_model)
    
    print("="*70)
    print("✓ Chatbot ready!\n")
    
    return chatbot


if __name__ == "__main__":
    chatbot = create_chatbot()
    chatbot.run_cli()
