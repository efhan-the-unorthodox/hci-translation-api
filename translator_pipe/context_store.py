from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import torch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

_VECTOR_CACHE: dict[Path, FAISS] = {}

from langchain_huggingface import HuggingFaceEmbeddings


def _get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Note: model_kwargs are passed to SentenceTransformer, which then has its own
    # model_kwargs passed to the underlying transformer model
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={
            "device": device,
        }
    )


def faiss_from_documents_batched(docs, embeddings, batch_size=32) -> FAISS:
    # Build from first batch, then add the rest incrementally
    vs = FAISS.from_documents(docs[:batch_size], embeddings)
    for i in range(batch_size, len(docs), batch_size):
        vs.add_documents(docs[i : i + batch_size])
    return vs


def with_retries(fn, tries=4, base_sleep=1.0):
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt == tries - 1:
                raise
            time.sleep(base_sleep * (2**attempt))
    raise last


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
