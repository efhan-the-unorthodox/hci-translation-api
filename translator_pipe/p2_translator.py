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


def _parse_numbered_list(content: str, expected_count: int) -> List[str]:
    """
    Parse numbered list format: "1. text\n2. text\n3. text"
    Falls back gracefully if parsing fails.

    Args:
        content: The response content to parse
        expected_count: Number of suggestions expected

    Returns:
        List of exactly expected_count suggestions
    """
    lines = content.strip().split('\n')
    suggestions = []

    for line in lines:
        line = line.strip()
        # Match patterns: "1.", "1)", "1 -", etc.
        if line and len(line) > 0:
            # Check if line starts with a digit
            first_char = line[0]
            if first_char.isdigit():
                # Remove number prefix (handles: "1.", "1)", "1 -", etc.)
                text = line
                # Try splitting by different delimiters
                if '.' in text:
                    text = text.split('.', 1)[-1].strip()
                elif ')' in text:
                    text = text.split(')', 1)[-1].strip()

                # Remove leading dash if present
                if text.startswith('-'):
                    text = text[1:].strip()

                if text:
                    suggestions.append(text)

    # Edge case handling: if no numbered list detected, treat entire response as one suggestion
    if len(suggestions) == 0 and content.strip():
        suggestions = [content.strip()]

    # Ensure we always return expected_count
    while len(suggestions) < expected_count:
        # Duplicate first suggestion if we have at least one
        if suggestions:
            suggestions.append(suggestions[0])
        else:
            # Fallback: empty strings
            suggestions.append("")

    # Truncate if we got more than expected
    return suggestions[:expected_count]


def translate_suggestions(
    input_sentence: str,
    source_lang: str,
    dest_lang: str,
    previous_sentence: Optional[str] = None,
    next_sentence: Optional[str] = None,
    context_chunks: Optional[List[str]] = None,
    num_suggestions: int = 3,
) -> List[str]:
    """
    Translate a single sentence using document context, returning multiple suggestions.
    Uses a numbered list format for reliable parsing.

    Args:
        input_sentence: The sentence to translate
        source_lang: Source language
        dest_lang: Target language
        previous_sentence: Previous sentence for context
        next_sentence: Next sentence for context
        context_chunks: Document context snippets
        num_suggestions: Number of suggestions to generate (default 3)

    Returns:
        List of translation suggestions
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a professional translator. Translate the user-provided "
                    "sentence accurately while preserving meaning, tone, and terminology. "
                    "Use the supplied document context to resolve ambiguity. "
                    f"Provide exactly {num_suggestions} different translation variations. "
                    "Return them as a numbered list (1. ... 2. ... 3. ...) with no additional text."
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

    # Use higher temperature for variation
    if not HF_API_KEY:
        raise ValueError("HF_API_KEY is not set in the environment.")
    client = ChatOpenAI(
        model=HF_MODEL,
        base_url=HF_BASE_URL,
        api_key=HF_API_KEY,
        temperature=0.4,  # Increased from 0.2 for more variation
    )

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

    return _parse_numbered_list(response.content.strip(), num_suggestions)


def generate_segment_alternatives(
    segment_text: str,
    full_sentence: str,
    position: int,
    language: str,
    num_alternatives: int = 3,
) -> List[str]:
    """
    Generate alternative phrasings for a specific segment within a sentence.

    Args:
        segment_text: The segment to rephrase
        full_sentence: The complete sentence for context
        position: Position of segment in sentence (0-based index)
        language: Target language (e.g., "English", "Chinese (Simplified)")
        num_alternatives: Number of alternatives to generate (default 3)

    Returns:
        List of alternative phrasings for the segment
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    f"You are a professional translator and editor. Generate {num_alternatives} "
                    "alternative phrasings for a specific segment within a sentence. "
                    "The alternatives should:\n"
                    "1. Preserve the original meaning\n"
                    "2. Maintain grammatical coherence with the rest of the sentence\n"
                    "3. Vary in style, formality, or word choice\n"
                    "4. Be natural and idiomatic\n\n"
                    f"Return exactly {num_alternatives} alternatives as a numbered list "
                    "(1. ... 2. ... 3. ...) with no additional text."
                ),
            ),
            (
                "user",
                (
                    "Language: {language}\n\n"
                    "Full sentence: {full_sentence}\n\n"
                    "Segment to rephrase: \"{segment_text}\" (position {position})\n\n"
                    f"Generate {num_alternatives} alternative phrasings for this segment "
                    "that work well in the context of the full sentence."
                ),
            ),
        ]
    )

    if not HF_API_KEY:
        raise ValueError("HF_API_KEY is not set in the environment.")

    client = ChatOpenAI(
        model=HF_MODEL,
        base_url=HF_BASE_URL,
        api_key=HF_API_KEY,
        temperature=0.5,  # Higher temperature for more variation
    )

    response = client.invoke(
        prompt.format_messages(
            language=language,
            full_sentence=full_sentence,
            segment_text=segment_text,
            position=position,
        )
    )

    # Reuse existing _parse_numbered_list helper
    return _parse_numbered_list(response.content.strip(), num_alternatives)


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
