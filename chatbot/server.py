#!/usr/bin/env python3
"""
FastAPI wrapper for the rule-based chatbot
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import uuid
from typing import Dict, List, Optional
import asyncio
import logging

from stage1_rulebot_v3 import RuleBasedChatbot, run_script_mode_from_json

# Initialize FastAPI app
app = FastAPI(title="Medical Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for chat sessions (use Redis/DB in production)
chat_sessions: Dict[str, Dict] = {}

@app.get("/")
async def root():
    return {"message": "Medical Chatbot API", "status": "healthy"}

@app.post("/chats")
async def create_chat():
    """Create a new chat session"""
    chat_id = str(uuid.uuid4())
    chat_sessions[chat_id] = {
        "chat_id": chat_id,
        "bot": RuleBasedChatbot(),
        "messages": []
    }
    return {"id": chat_id, "message": "Chat session created"}

@app.post("/chats/{chat_id}")
async def send_chat_message(chat_id: str, request: Dict):
    """Send a message to an existing chat session"""
    if chat_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    message = request.get("message", "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    chat_session = chat_sessions[chat_id]
    bot = chat_session["bot"]
    
    # Store user message
    chat_session["messages"].append({"role": "user", "content": message})
    
    # Process the message
    if bot.state.phase == "diagnose":
        response = bot.process_confirmation(message)
    else:
        response = bot.process_user_input(message)
    
    # Store bot response
    chat_session["messages"].append({"role": "assistant", "content": response})
    
    return {
        "chat_id": chat_id,
        "response": response,
        "phase": bot.state.phase,
        "collected_symptoms": bot.state.positives()
    }

@app.post("/chats/{chat_id}/stream")
async def send_chat_message_stream(chat_id: str, request: Dict):
    """Streaming version for real-time responses"""
    if chat_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    message = request.get("message", "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    chat_session = chat_sessions[chat_id]
    bot = chat_session["bot"]
    
    async def generate():
        if bot.state.phase == "diagnose":
            response_text = bot.process_confirmation(message)
        else:
            response_text = bot.process_user_input(message)

        # Store bot response for history
        chat_session["messages"].append({"role": "assistant", "content": response_text})

        # Split response logically on intended line breaks
        for line in response_text.split("\n"):
            yield f"data: {json.dumps({'content': line})}\n\n"
        yield "data: [DONE]\n\n"


@app.get("/chats/{chat_id}")
async def get_chat_history(chat_id: str):
    """Get chat history"""
    if chat_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    return {
        "chat_id": chat_id,
        "messages": chat_sessions[chat_id]["messages"],
        "collected_symptoms": chat_sessions[chat_id]["bot"].state.positives()
    }

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat session"""
    if chat_id in chat_sessions:
        del chat_sessions[chat_id]
    return {"message": "Chat session deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)