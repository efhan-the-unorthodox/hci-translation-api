"""
LangChain-powered translation helpers using the HuggingFace OpenAI-compatible API.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_BASE_URL = "https://router.huggingface.co/v1"
HF_MODEL = "openai/gpt-oss-20b"


def _get_chat_client() -> ChatOpenAI:
    if not HF_API_KEY:
        raise ValueError("HF_API_KEY is not set in the environment.")
    return ChatOpenAI(
        model=HF_MODEL,
        base_url=HF_BASE_URL,
        api_key=HF_API_KEY,
        temperature=0.2,
    )


def _build_context_block(context_chunks: Optional[Iterable[str]]) -> str:
    if not context_chunks:
        return "No additional document context provided."
    joined = "\n\n".join(f"- {chunk}" for chunk in context_chunks)
    return f"Document context snippets:\n{joined}"


def translate(
    input_sentence: str,
    source_lang: str,
    dest_lang: str,
    previous_sentence: Optional[str] = None,
    next_sentence: Optional[str] = None,
    context_chunks: Optional[List[str]] = None,
) -> str:
    """Translate a single sentence using document context."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a professional translator. Translate the user-provided "
                    "sentence accurately while preserving meaning, tone, and terminology. "
                    "Use the supplied document context to resolve ambiguity. "
                    "Return only the translated sentence."
                ),
            ),
            (
                "user",
                (
                    "Source language: {source_lang}\n"
                    "Target language: {dest_lang}\n\n"
                    "{context_block}\n\n"
                    "Previous sentence: {previous_sentence}\n"
                    "Current sentence: {input_sentence}\n"
                    "Next sentence: {next_sentence}\n"
                ),
            ),
        ]
    )

    client = _get_chat_client()
    context_block = _build_context_block(context_chunks)
    response = client.invoke(
        prompt.format_messages(
            source_lang=source_lang,
            dest_lang=dest_lang,
            context_block=context_block,
            previous_sentence=previous_sentence or "(none)",
            input_sentence=input_sentence,
            next_sentence=next_sentence or "(none)",
        )
    )
    return response.content.strip()


def translate_para(
    paragraph: str,
    source_lang: str,
    dest_lang: str,
    context_chunks: Optional[List[str]] = None,
) -> str:
    """Translate a paragraph while using document context."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a professional translator. Translate the paragraph accurately "
                    "while preserving meaning, tone, and terminology. "
                    "Use the supplied document context to resolve ambiguity. "
                    "Return only the translated paragraph."
                ),
            ),
            (
                "user",
                (
                    "Source language: {source_lang}\n"
                    "Target language: {dest_lang}\n\n"
                    "{context_block}\n\n"
                    "Paragraph:\n{paragraph}"
                ),
            ),
        ]
    )

    client = _get_chat_client()
    context_block = _build_context_block(context_chunks)
    response = client.invoke(
        prompt.format_messages(
            source_lang=source_lang,
            dest_lang=dest_lang,
            context_block=context_block,
            paragraph=paragraph,
        )
    )
    return response.content.strip()
