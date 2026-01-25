"""
Alternate phrasing generation using LangChain and HuggingFace OpenAI-compatible API.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

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
        temperature=0.4,
    )


def _build_context_block(context_chunks: Optional[List[str]]) -> str:
    if not context_chunks:
        return "No additional document context provided."
    joined = "\n\n".join(f"- {chunk}" for chunk in context_chunks)
    return f"Document context snippets:\n{joined}"


def generate_alternate_phrasing(
    phrase: str,
    current_sentence: str,
    previous_sentence: Optional[str] = None,
    next_sentence: Optional[str] = None,
    context_chunks: Optional[List[str]] = None,
    suggestion_count: int = 3,
) -> List[str]:
    """
    Generate alternate phrasings for a translated phrase, using sentence context
    and document context. Returns a list of exactly suggestion_count strings.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a language assistant that suggests alternate phrasings "
                    "for a translated phrase. Keep the meaning faithful to the original "
                    "sentence and use the document context to preserve terminology. "
                    "Return only a JSON array of strings with no extra text."
                ),
            ),
            (
                "user",
                (
                    "{context_block}\n\n"
                    "Previous sentence: {previous_sentence}\n"
                    "Current sentence: {current_sentence}\n"
                    "Next sentence: {next_sentence}\n\n"
                    "Phrase to rephrase: {phrase}\n\n"
                    "Provide exactly {suggestion_count} alternative phrasings."
                ),
            ),
        ]
    )

    client = _get_chat_client()
    context_block = _build_context_block(context_chunks)
    response = client.invoke(
        prompt.format_messages(
            context_block=context_block,
            previous_sentence=previous_sentence or "(none)",
            current_sentence=current_sentence,
            next_sentence=next_sentence or "(none)",
            phrase=phrase,
            suggestion_count=suggestion_count,
        )
    )

    content = response.content.strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            alternatives = [str(item).strip() for item in parsed if str(item).strip()]
        else:
            alternatives = []
    except json.JSONDecodeError:
        alternatives = [line.strip("- ").strip() for line in content.splitlines() if line.strip()]

    if len(alternatives) >= suggestion_count:
        return alternatives[:suggestion_count]
    if not alternatives:
        alternatives = [phrase]
    while len(alternatives) < suggestion_count:
        alternatives.append(phrase)
    return alternatives[:suggestion_count]
