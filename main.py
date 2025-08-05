from typing import Union
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from translator_pipe.p1_translator import *
from translator_pipe.p2_chunking import *
from translator_pipe.p4_alt_builder import *
from translator_pipe.p1_5_sentence_segmt import *
from fastapi.middleware.cors import CORSMiddleware
import ast
from datetime import datetime
import json

app = FastAPI()

# In-memory log storage
logs_storage = []

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected clients
                self.active_connections.remove(connection)

manager = ConnectionManager()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # allow only your React app origins
    allow_credentials=True,
    allow_methods=["*"],         # allow GET, POST, PUT, DELETE, OPTIONS, etc.
    allow_headers=["*"],         # allow all headers
)


@app.get("/")
def read_root():
    
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


"""
init_translation 
This post request receives the raw source sentence from the frontend and will send it through the first two stages of 
the translation pipeline. It will `translate` to English. Then with the bpe tokens from the OpenAI ChatCompletion, 
it will then pass through spacy and do the necessary chunking process. See it for yourself in the pipeline.
"""
class TranslationInput(BaseModel):
    input_sent:str
    source_lang: str
    
@app.post("/translate")
def init_translation(transInput: TranslationInput):
    tokens, logrows = translate(transInput.input_sent, transl_frm=transInput.source_lang)
    chunks, plaintext = chunk_sentence(tokens)
    return {"chunks": chunks, "plaintext": plaintext}

class ParaInput(BaseModel):
    input_para: str
    source_lang :str

@app.post("/translate_para")
def para_translation(paraInput:ParaInput):
    tsl_para = translate_para(paraInput.input_para, transl_frm=paraInput.source_lang)
    sentences = segment_para(tsl_para)
    
    return sentences
    
class SentInput(BaseModel):
    input_sent:str 
@app.post("/sentence_chunking")
def generate_sent_chunks(sentInput:SentInput):
    chunks = chunk_sentence_2(sentInput.input_sent)
    return chunks
    
class AlternativeInput(BaseModel):
    chunk_text: str
    whole_sent:str
    
@app.post("/alternate_chunking")
def generate_alternatives(data: AlternativeInput):
    alt_resp = generate_alternate_phrasing(data.chunk_text, data.whole_sent)
    if isinstance(alt_resp, str):
        parsed_list = ast.literal_eval(alt_resp)
    return parsed_list

# Logging endpoint for frontend logs
class LogData(BaseModel):
    level: str
    message: str
    payload: list
    ts: int

@app.post("/logs")
async def receive_logs(log_data: LogData):
    """
    Receive logs from the frontend and store them in memory
    """
    timestamp = datetime.fromtimestamp(log_data.ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
    
    # Create log entry
    log_entry = {
        "level": log_data.level,
        "message": log_data.message,
        "payload": log_data.payload,
        "ts": log_data.ts,
        "timestamp": timestamp
    }
    
    # Store in memory
    logs_storage.append(log_entry)
    
    # Keep only last 1000 logs to prevent memory issues
    if len(logs_storage) > 1000:
        logs_storage.pop(0)
    
    # Broadcast new log to all connected WebSocket clients
    await manager.broadcast(json.dumps({
        "type": "new_log",
        "log": log_entry
    }))
    
    # Print to console for debugging
    print(f"\n{'='*60}")
    print(f"[{timestamp}] {log_data.level.upper()}: {log_data.message}")
    print(f"{'='*60}")
    
    if log_data.payload and len(log_data.payload) > 0:
        payload = log_data.payload[0]  # Get the first payload item
        
        if log_data.message == "Translation started":
            print(f"User Input: {payload.get('userInput', 'N/A')}")
            print(f"Source Language: {payload.get('sourceLang', 'N/A')}")
            
        elif log_data.message == "Translation completed":
            print(f"Original Input: {payload.get('originalInput', 'N/A')}")
            print(f"Source Language: {payload.get('sourceLang', 'N/A')}")
            print(f"Translated Sentences ({len(payload.get('translatedSentences', []))}):")
            for i, sentence in enumerate(payload.get('translatedSentences', [])):
                print(f"  {i+1}. {sentence}")
            print(f"Selected Chunks ({len(payload.get('selectedChunks', []))}):")
            for i, chunk in enumerate(payload.get('selectedChunks', [])):
                print(f"  {i+1}. Text: {chunk.get('text', 'N/A')}")
                if chunk.get('alternatives'):
                    print(f"     Alternatives: {', '.join(chunk.get('alternatives', []))}")
                    
        elif log_data.message == "Chunk alternative selected":
            print(f"Original Chunk: {payload.get('NewChunk', 'N/A')}")
            print(f"Selected Alternative: {payload.get('OldChunk', 'N/A')}")
            print(f"Sentence Index: {payload.get('sentenceIndex', 'N/A')}")
            print(f"Chunk Index: {payload.get('chunkIndex', 'N/A')}")
            
        elif log_data.message == "Complete sentence updated":
            print(f"COMPLETE SENTENCE UPDATED")
            print(f"Sentence Index: {payload.get('sentenceIndex', 'N/A')}")
            print(f"New Complete Sentence: {payload.get('newSentence', 'N/A')}")
            
        else:
            print(f"Payload: {payload}")
    
    print(f"{'='*60}\n")
    
    return {"status": "logged"}

@app.get("/logs")
def get_logs():
    """
    Get all stored logs
    """
    return logs_storage

@app.delete("/logs")
def clear_logs():
    """
    Clear all stored logs
    """
    global logs_storage
    logs_storage = []
    return {"status": "cleared"}

@app.get("/logs/export")
def export_logs():
    """
    Export logs as JSON
    """
    return logs_storage

# WebSocket endpoint for real-time log updates
@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial logs to the client
        await websocket.send_text(json.dumps({
            "type": "initial_logs",
            "logs": logs_storage
        }))
        
        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            # You can handle client messages here if needed
            # For now, we just keep the connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)