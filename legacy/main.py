# main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary chat store
chats = {}

@app.post("/chats")
async def create_chat():
    chat_id = str(uuid.uuid4())
    chats[chat_id] = []
    return {"id": chat_id}

@app.post("/chats/{chat_id}")
async def send_chat_message(chat_id: str, request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    chats[chat_id].append({"role": "user", "content": user_message})

    async def event_stream():
        # Simulate streaming chunks (like OpenAI)
        text_response = f"I think you may be experiencing: {user_message}"
        for word in text_response.split():
            yield f"data: {word} \n\n"
            await asyncio.sleep(0.15)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
