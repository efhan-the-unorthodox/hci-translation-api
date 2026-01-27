from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_BASE_URL = "https://router.huggingface.co/v1"
HF_EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

_VECTOR_CACHE: dict[Path, FAISS] = {}


def _get_embeddings() -> OpenAIEmbeddings:
    if not HF_API_KEY:
        raise ValueError("HF_API_KEY is not set in the environment.")
    return OpenAIEmbeddings(
        model=HF_EMBEDDING_MODEL,
        base_url=HF_BASE_URL,
        api_key=HF_API_KEY,
    )


def build_project_index(
    project_dir: Path,
    document_text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> Path:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = splitter.split_documents([Document(page_content=document_text)])
    embeddings = _get_embeddings()
    try:
        vector_store = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        print("Failed to build FAISS vector store:", e)
        raise
    print("LINE 48", vector_store)

    index_dir = project_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    print("LINE 52", index_dir)
    vector_store.save_local(str(index_dir))
    _VECTOR_CACHE[index_dir] = vector_store
    return index_dir


def _load_project_index(project_dir: Path) -> Optional[FAISS]:
    index_dir = project_dir / "index"
    if index_dir in _VECTOR_CACHE:
        return _VECTOR_CACHE[index_dir]
    if not index_dir.exists():
        return None
    embeddings = _get_embeddings()
    vector_store = FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )
    _VECTOR_CACHE[index_dir] = vector_store
    return vector_store


def warm_project_index(project_dir: Path) -> bool:
    return _load_project_index(project_dir) is not None


def retrieve_context(project_dir: Path, query: str, k: int = 4) -> List[str]:
    vector_store = _load_project_index(project_dir)
    if not vector_store or not query:
        return []
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]
