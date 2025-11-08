// utils.js
import { EventSourceParserStream } from 'eventsource-parser/stream';

export async function* parseSSEStream(stream) {
  const sseReader = stream
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(new EventSourceParserStream())
    .getReader();
  
  while (true) {
    const { done, value } = await sseReader.read();
    if (done) break;

    console.log("hunk:", JSON.stringify(value));
    
    // Check for completion signal
    if (value.data === '[DONE]') break;
    
    try {
      // Parse the JSON to extract content
      const parsed = JSON.parse(value.data);
      yield parsed.content;  // Extract the text content
    } catch (error) {
      // If it's not JSON, use it as plain text
      yield value.data;
    }
  }
}