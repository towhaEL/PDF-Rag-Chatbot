# backend/app/main.py
from fastapi import FastAPI, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import shutil

from app.services.rag_pipeline import ingest_pdf, answer_query, get_history_for_session

app = FastAPI()

# Allow frontend requests (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None  # optional client-provided session id

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), x_session_id: Optional[str] = Header(None)):
    """
    Upload PDF and ingest into vector DB.
    The client can pass a header `X-Session-Id` (optional).
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ingest into vector DB (creates/reuses global vector DB)
    ingest_pdf(file_path)

    return {"status": "success", "message": f"{file.filename} processed"}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Ask a question. Include optional session_id to maintain per-user chat history.
    """
    session_id = request.session_id or "default"
    answer = answer_query(request.question, session_id=session_id)
    return {"answer": answer['answer'], "sources": answer['sources']}

@app.get("/history")
async def get_history(session_id: Optional[str] = None):
    """
    Fetch chat history for a session (useful if frontend wants authoritative history).
    """
    sid = session_id or "default"
    return {"session_id": sid, "history": get_history_for_session(sid)}
