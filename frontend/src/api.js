const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function createChat() {
  const res = await fetch(BASE_URL + '/chats', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data;
}

async function sendChatMessage(chatId, message) {
  const res = await fetch(BASE_URL + `/chats/${chatId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message })
  });
  
  if (!res.ok) {
    return Promise.reject({ status: res.status, data: await res.json() });
  }
  
  const data = await res.json();
  
  // Convert the response to a stream-like format that your frontend expects
  const stream = new ReadableStream({
    start(controller) {
      const text = data.response;
      
      // Send the entire text at once, preserving all formatting
      controller.enqueue(
        new TextEncoder().encode(`data: ${JSON.stringify({ content: text })}\n\n`)
      );
      controller.enqueue(
        new TextEncoder().encode('data: [DONE]\n\n')
      );
      controller.close();
    }
  });
  
  return stream;
}

// Alternative: Use the actual streaming endpoint
async function sendChatMessageStream(chatId, message) {
  const res = await fetch(BASE_URL + `/chats/${chatId}/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message })
  });
  
  if (!res.ok) {
    return Promise.reject({ status: res.status, data: await res.json() });
  }
  
  return res.body;
}

export default {
  createChat,
  sendChatMessage,
  sendChatMessageStream
};