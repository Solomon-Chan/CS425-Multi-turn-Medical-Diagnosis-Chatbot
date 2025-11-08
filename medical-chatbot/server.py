"""
FastAPI Server for Medical Chatbot

Provides REST API endpoints for the medical symptom assessment chatbot.
Supports both regular and streaming responses.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import uuid
from typing import Dict, Optional
import asyncio

from chatbot import MedicalChatbot, ChatState
from symptom_extractor import MLSymptomExtractor
from disease_identifier import DiseaseIdentifier

# Initialize FastAPI app
app = FastAPI(
    title="Medical Chatbot API",
    version="2.0.0",
    description="AI-powered medical symptom assessment with confirmation flow"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (initialized once at startup)
symptom_extractor = None
disease_identifier = None

# In-memory storage for chat sessions
chat_sessions: Dict[str, MedicalChatbot] = {}


class MessageRequest(BaseModel):
    """Request model for sending messages."""
    message: str


class MessageResponse(BaseModel):
    """Response model for messages."""
    chat_id: str
    response: str
    phase: str
    confirmed_symptoms: list
    diagnosis: Optional[dict] = None


@app.on_event("startup")
async def startup_event():
    """Initialize ML components on startup."""
    global symptom_extractor, disease_identifier
    
    print("\n" + "="*70)
    print("Starting Medical Chatbot API")
    print("="*70 + "\n")
    
    print("Loading ML components...")
    
    # Initialize symptom extractor
    symptom_extractor = MLSymptomExtractor()
    print()
    
    # Initialize disease identifier
    disease_identifier = DiseaseIdentifier()
    print()
    
    print("="*70)
    print("✓ API ready to accept requests")
    print("="*70 + "\n")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Medical Chatbot API",
        "version": "2.0.0",
        "status": "healthy"
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "components": {
            "symptom_extractor": symptom_extractor is not None,
            "disease_identifier": disease_identifier is not None
        },
        "active_sessions": len(chat_sessions)
    }


@app.post("/chats")
async def create_chat():
    """
    Create a new chat session.
    
    Returns:
        {"id": "chat-uuid"}
    """
    chat_id = str(uuid.uuid4())
    
    # Create new chatbot instance for this session
    chatbot = MedicalChatbot(symptom_extractor, disease_identifier)
    chat_sessions[chat_id] = chatbot
    
    return {
        "id": chat_id,
        "message": "Chat session created"
    }


@app.post("/chats/{chat_id}", response_model=MessageResponse)
async def send_message(chat_id: str, request: MessageRequest):
    """
    Send a message to a chat session.
    
    Args:
        chat_id: Chat session ID
        request: Message request with 'message' field
        
    Returns:
        MessageResponse with bot's response and state
    """
    if chat_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    chatbot = chat_sessions[chat_id]
    
    # Process message
    response = chatbot.process_message(request.message)
    
    # Build response
    return MessageResponse(
        chat_id=chat_id,
        response=response,
        phase=chatbot.state.phase.value,
        confirmed_symptoms=chatbot.state.confirmed_symptoms,
        diagnosis=chatbot.state.diagnosis_result
    )


@app.post("/chats/{chat_id}/stream")
async def send_message_stream(chat_id: str, request: MessageRequest):
    """
    Send a message and get streaming response.
    
    Args:
        chat_id: Chat session ID
        request: Message request with 'message' field
        
    Returns:
        Server-Sent Events stream
    """
    if chat_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    chatbot = chat_sessions[chat_id]
    
    async def generate():
        # Process message
        response = chatbot.process_message(request.message)
        
        # Stream response word by word
        words = response.split()
        for word in words:
            yield f"data: {json.dumps({'content': word + ' '})}\n\n"
            await asyncio.sleep(0.05)  # Slight delay for streaming effect
        
        # Send done signal
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/chats/{chat_id}")
async def get_chat_state(chat_id: str):
    """
    Get current state of a chat session.
    
    Args:
        chat_id: Chat session ID
        
    Returns:
        Current chat state
    """
    if chat_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    chatbot = chat_sessions[chat_id]
    state = chatbot.state
    
    return {
        "chat_id": chat_id,
        "phase": state.phase.value,
        "confirmed_symptoms": state.confirmed_symptoms,
        "symptom_count": state.symptom_count(),
        "turn": state.turn,
        "diagnosis": state.diagnosis_result
    }


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """
    Delete a chat session.
    
    Args:
        chat_id: Chat session ID
        
    Returns:
        Success message
    """
    if chat_id in chat_sessions:
        del chat_sessions[chat_id]
    
    return {"message": "Chat session deleted"}


@app.post("/chats/{chat_id}/restart")
async def restart_chat(chat_id: str):
    """
    Restart a chat session (reset state).
    
    Args:
        chat_id: Chat session ID
        
    Returns:
        Success message
    """
    if chat_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    chatbot = chat_sessions[chat_id]
    chatbot.state.reset()
    
    return {
        "message": "Chat session restarted",
        "welcome": chatbot.welcome()
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("Starting Medical Chatbot Server")
    print("="*70)
    print("\nServer will start on: http://0.0.0.0:8000")
    print("API docs available at: http://0.0.0.0:8000/docs")
    print("\nPress Ctrl+C to stop")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
