from typing import Union
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from pydantic import BaseModel
from translator_pipe.p0_preprocessing import *
from translator_pipe.p2_chunking import *
from translator_pipe.p4_alt_builder import *
from translator_pipe.p1_5_sentence_segmt import *
from fastapi.middleware.cors import CORSMiddleware
import ast
from datetime import datetime
import json
from fastapi import UploadFile, File, Form
import csv
import io
import os
from pathlib import Path

# Paths for project data storage
PROJECTS_JSON_PATH = Path(__file__).parent / "projects.json"
PROJECTS_FOLDER_PATH = Path(__file__).parent / "projects"

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

origins = ["https://main.d3f1dka39jgvuq.amplifyapp.com/", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # allow only your React app origins
    allow_credentials=False,
    allow_methods=["*"],  # allow GET, POST, PUT, DELETE, OPTIONS, etc.
    allow_headers=["*"],  # allow all headers
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


# class TranslationInput(BaseModel):
#     input_sent: str
#     source_lang: str
#
#
# @app.post("/translate")
# def init_translation(transInput: TranslationInput):
#     """
#     Translation endpoint temporarily disabled while the translation
#     pipeline is being reimplemented. This endpoint previously called
#     `translate()` which depended on the OpenAI API.
#     """
#     tokens, logrows = translate(
#         transInput.input_sent, transl_frm=transInput.source_lang
#     )
#     chunks, plaintext = chunk_sentence(tokens)
#     return {"chunks": chunks, "plaintext": plaintext}


class ParaInput(BaseModel):
    source_lang: str


class ProjectInfo(BaseModel):
    project_id: str
    projectName: str
    dateCreated: str
    source_lang: str
    dest_lang: str
    documentName: str
    dateCreated: str


@app.post("/preprocessing")
async def text_preprocessing(
    project_id: str = Form(...),
    projectName: str = Form(...),
    source_lang: str = Form(...),
    dest_lang: str = Form(...),
    documentName: str = Form(...),
    dateCreated: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Accept a single uploaded file (plain text, PDF, or .docx Word document).
    Extracts text content, performs sentence segmentation, and returns the results.
    """
    uploaded: UploadFile = file
    content = await uploaded.read()
    filename = (uploaded.filename or "").lower()
    content_type = (uploaded.content_type or "").lower()

    print(project_id)

    # Determine file extension
    ext = filename.rsplit(".", 1)[-1] if "." in filename else ""

    # Detect file type
    is_pdf = ext == "pdf" or "pdf" in content_type
    is_docx = (
        ext == "docx"
        or "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        in content_type
    )
    is_old_doc = ext == "doc" or (
        "application/msword" in content_type and ext != "docx"
    )
    is_text = content_type.startswith("text") or ext in ("txt", "md")

    # Reject old .doc files (only .docx supported)
    if is_old_doc:
        return {
            "filename": filename,
            "format": "unsupported",
            "error": "Legacy .doc files are not supported. Please convert to .docx format.",
            "source_lang": source_lang,
        }

    # Extract text based on file type
    extracted_text = ""
    error_msg = ""
    file_format = "unknown"

    if is_text:
        file_format = "text"
        extracted_text, error_msg = extract_text_from_txt(content)
    elif is_pdf:
        file_format = "pdf"
        extracted_text, error_msg = extract_text_from_pdf(content)
    elif is_docx:
        file_format = "docx"
        extracted_text, error_msg = extract_text_from_docx(content)
    else:
        return {
            "filename": filename,
            "format": "unknown",
            "error": "Unsupported file format. Please upload a .txt, .md, .pdf, or .docx file.",
            "source_lang": source_lang,
        }

    # Handle extraction errors
    if error_msg:
        return {
            "filename": filename,
            "format": file_format,
            "error": error_msg,
            "source_lang": source_lang,
        }

    # Check if we got any text
    if not extracted_text.strip():
        return {
            "filename": filename,
            "format": file_format,
            "error": "No text content could be extracted from the file.",
            "source_lang": source_lang,
        }

    # Perform sentence segmentation
    sentences = segment_para(extracted_text, lang=source_lang)

    # Print segmented sentences for debugging
    print(f"\n{'='*60}")
    print(f"PREPROCESSING: {filename}")
    print(f"Format: {file_format} | Source Language: {source_lang} ")
    print(f"{'='*60}")
    print(f"Extracted {len(sentences)} sentences:")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent[:100]}{'...' if len(sent) > 100 else ''}")
    print(f"{'='*60}\n")

    # --- Save project data ---
    # 1. Append project info to projects.json
    project_record = {
        "id": project_id,
        "name": projectName,
        "dateCreated": dateCreated,
        "sourceLanguage": source_lang,
        "destinationLanguage": dest_lang,
        "documentName": documentName,
        "termBaseName": "",
        "lastEdit": dateCreated,
    }

    # Read existing projects, append new one, write back
    projects_list = []
    if PROJECTS_JSON_PATH.exists():
        with open(PROJECTS_JSON_PATH, "r", encoding="utf-8") as f:
            try:
                projects_list = json.load(f)
            except json.JSONDecodeError:
                projects_list = []

    projects_list.append(project_record)

    with open(PROJECTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(projects_list, f, indent=4, ensure_ascii=False)

    # 2. Create sentences file in projects folder
    PROJECTS_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

    sentences_data = [
        {
            "id": idx,
            "sourceText": sent,
            "translation": "",
            "isComplete": False,
        }
        for idx, sent in enumerate(sentences)
    ]

    project_file_path = PROJECTS_FOLDER_PATH / f"{project_id}.json"
    with open(project_file_path, "w", encoding="utf-8") as f:
        json.dump(sentences_data, f, indent=4, ensure_ascii=False)

    print(f"Project saved: {project_id} with {len(sentences)} sentences")

    return sentences_data


@app.post("/projects")
def get_projects():
    """
    Return stored project metadata from projects.json.
    """
    if not PROJECTS_JSON_PATH.exists():
        return []

    with open(PROJECTS_JSON_PATH, "r", encoding="utf-8") as f:
        try:
            projects_list = json.load(f)
        except json.JSONDecodeError:
            return []

    return projects_list


class ProjId(BaseModel):
    id: str


@app.get("/sentences")
def get_sentences(id: str):
    """
    Return stored sentences for a given project id (404 if not found).
    """
    project_file = PROJECTS_FOLDER_PATH / f"{id}.json"
    if not project_file.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        with open(project_file, "r", encoding="utf-8") as f:
            sentences = json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Project file is corrupted")

    return sentences


## Translation-related endpoint temporarily disabled
## @app.post("/translate_para")
## def para_translation(paraInput: ParaInput):
##     tsl_para = translate_para(paraInput.input_para, transl_frm=paraInput.source_lang)
##     sentences = segment_para(tsl_para)
##     return sentences


class SentInput(BaseModel):
    input_sent: str


@app.post("/sentence_chunking")
def generate_sent_chunks(sentInput: SentInput):
    chunks = chunk_sentence_2(sentInput.input_sent)
    return chunks


class AlternativeInput(BaseModel):
    chunk_text: str
    whole_sent: str


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
    timestamp = datetime.fromtimestamp(log_data.ts / 1000).strftime("%Y-%m-%d %H:%M:%S")

    # Create log entry
    log_entry = {
        "level": log_data.level,
        "message": log_data.message,
        "payload": log_data.payload,
        "ts": log_data.ts,
        "timestamp": timestamp,
    }

    # Store in memory
    logs_storage.append(log_entry)

    # Keep only last 1000 logs to prevent memory issues
    if len(logs_storage) > 1000:
        logs_storage.pop(0)

    # Broadcast new log to all connected WebSocket clients
    await manager.broadcast(json.dumps({"type": "new_log", "log": log_entry}))

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
            print(
                f"Translated Sentences ({len(payload.get('translatedSentences', []))}):"
            )
            for i, sentence in enumerate(payload.get("translatedSentences", [])):
                print(f"  {i+1}. {sentence}")
            print(f"Selected Chunks ({len(payload.get('selectedChunks', []))}):")
            for i, chunk in enumerate(payload.get("selectedChunks", [])):
                print(f"  {i+1}. Text: {chunk.get('text', 'N/A')}")
                if chunk.get("alternatives"):
                    print(
                        f"     Alternatives: {', '.join(chunk.get('alternatives', []))}"
                    )

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
        await websocket.send_text(
            json.dumps({"type": "initial_logs", "logs": logs_storage})
        )

        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            # You can handle client messages here if needed
            # For now, we just keep the connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)
