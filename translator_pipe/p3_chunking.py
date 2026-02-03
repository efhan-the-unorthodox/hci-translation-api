from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
import spacy.tokens
import logging

# ------------------------------------------------------------------
# 1.  Load a lightweight English pipeline (no NER, but parser needed)
# ------------------------------------------------------------------
# For quick prototyping `en_core_web_sm` is enough.
# Swap to 'en_core_web_trf' later for better accuracy.
_NLP_CACHE: dict[str, spacy.language.Language] = {}


def _get_spacy_pipeline(lang: str) -> spacy.language.Language:
    lang = (lang or "en").lower()
    if lang.startswith("zh"):
        model = "zh_core_web_sm"
    elif lang.startswith("ja"):
        model = "ja_core_news_sm"
    else:
        model = "en_core_web_sm"

    if model not in _NLP_CACHE:
        nlp = spacy.load(model, disable=["ner", "lemmatizer"])
        nlp.enable_pipe("senter")
        _NLP_CACHE[model] = nlp

    return _NLP_CACHE[model]


_nlp = _get_spacy_pipeline("en")

# Initialize the Matcher once (do this at module top)
_matcher = Matcher(_nlp.vocab)

# Verb Phrase Matching
_matcher.add("VP", [[{"POS": "AUX", "OP": "*"}, {"POS": "VERB", "OP": "+"}]])
_matcher.add("OTHER", [[{"POS": "AUX", "OP": "*"}, {"POS": "PART", "OP": "+"}]])


# ------------------------------------------------------------------
# 2.  Helpers:  BPE tokens  →  plain text + char-level spans
# Essentially, what this thing is doing is called 'Detokenization'
# ------------------------------------------------------------------
def _tokens_to_text_and_char_spans(
    bpt_tokens: List[str],
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Re-join GPT BPE tokens (with leading 'Ġ' for spaces) into plain text.
    Return the text *and* a list mapping each token to its (start, end) char span.
    """
    text_parts: List[str] = []
    char_spans: List[Tuple[int, int]] = []

    cursor = 0

    for tok in bpt_tokens:
        surface = tok
        start = (
            cursor  # Set the starting point of the index where the token's text begins
        )
        text_parts.append(
            surface
        )  # Append the token's characters to text_parts i.e. this is one token that has been normalised
        cursor += len(
            surface
        )  # Increment the cursor by the length of the token that was just examined
        end = cursor  # Set the end index of the last character in the token
        char_spans.append(
            (start, end)
        )  # append the range of the start and end char index of the token to the char_spans

    plain_text = "".join(text_parts)

    return plain_text, char_spans


def add_single_token_chunks(doc: spacy.tokens.Doc, spans: List[Tuple[str, int, int]]):
    """
    `spans` is a list of (label, start_i, end_i) where start_i/end_i are token indices
    for NP and VP chunks.  Returns a **new list** that also includes
    1-token chunks ("TOK") filling every gap.
    """

    # 1) Build a boolean mask of covered tokens
    covered = [False] * len(doc)
    for _, start, end in spans:
        for i in range(start, end):  # end exclusive
            covered[i] = True

    # 2) Walk through tokens to add single-token gaps
    final = spans[:]  # copy existing NP/VP spans
    for i, tok in enumerate(doc):
        if not covered[i]:
            final.append(("TOK", i, i + 1))  # a 1-token span

    # 3) Sort everything in reading order
    final.sort(key=lambda t: t[1])  # by start token index

    return final


def _tokens_to_plaintext(oai_bpe_tokens: List[str]):
    """
    Re-join GPT BPE tokens into plain text.
    """
    text_parts: List[str] = []

    for tok in oai_bpe_tokens:
        text_parts.append(tok)

    plain_text = "".join(text_parts)

    return plain_text


# ------------------------------------------------------------------
# 3.  Map (start_char, end_char) → (start_token_idx, end_token_idx)
# ------------------------------------------------------------------
def _char_span_to_token_span(
    start_c: int,
    end_c: int,
    token_char_spans: List[Tuple[int, int]],
) -> Tuple[int, int]:
    start_tok = end_tok = None
    for idx, (t_start, t_end) in enumerate(token_char_spans):
        if t_end <= start_c:
            continue  # token before the span
        if start_tok is None:
            start_tok = idx
        if t_start < end_c:
            end_tok = idx  # last token whose start < end_c
        else:
            break
    return start_tok, end_tok


# ------------------------------------------------------------------
# 4.  Chunk extractor (NPs + VPs)
# ------------------------------------------------------------------
def _extract_chunks(doc: spacy.tokens.Doc):
    """
    Extract individual tokens as spans (including punctuation).
    Each token becomes its own span with its POS tag as the label.
    This approach works universally across all languages.
    """
    token_spans = [
        spacy.tokens.Span(doc, tok.i, tok.i + 1, label=tok.pos_)
        for tok in doc
    ]

    return token_spans


def chunk_sentence(bpe_tokens: List[str]) -> Tuple[List[Dict], str]:
    """
    Parameters
    ----------
    bpe_tokens : List[str]
        The token strings exactly as returned in
        `choice.logprobs.content[i]["token"]`.

    Returns
    -------
    chunks : List[dict]
        Each dict has keys: id, label, text, span_tokens [start, end]
    plain_text : str
        The reconstructed sentence (for debugging / display)
    """
    plain_text = _tokens_to_plaintext(bpe_tokens)
    doc = _nlp(plain_text)

    chunks = _extract_chunks(doc)  # Chunk the entire freaking document

    results = []
    for idx, span in enumerate(chunks):
        results.append(
            {
                "id": idx,
                "label": span.label_ if span.label_ else "OTHER",
                "text": span.text,
                "tok_range": [
                    span.start,
                    span.end,
                ],  # REMEMBER, the *end token* for any span from spaCy is exclusive when deriving the actual token text from the doc
            }
        )

    return results, plain_text


def chunk_sentence_2(sentence: str, lang: Optional[str] = "en") -> List[Dict]:
    """
    Parameters
    ----------
    sentence : str
        The original sentence
    lang : str, optional
        Language code (e.g. "en", "zh"), defaults to "en"
    Returns
    -------
    chunks : List[dict]
        Each dict has keys: id, label, text, tok_range [start, end]
    """
    nlp = _get_spacy_pipeline(lang or "en")
    doc = nlp(sentence)

    chunks = _extract_chunks(doc)  # Chunk the entire freaking document

    results = []
    for idx, span in enumerate(chunks):
        results.append(
            {
                "id": idx,
                "label": span.label_ if span.label_ else "OTHER",
                "text": span.text,
                "tok_range": [
                    span.start,
                    span.end,
                ],  # REMEMBER, the *end token* for any span from spaCy is exclusive when deriving the actual token text from the doc
            }
        )

    return results


def tokenize_sentence(sentence: str, lang: Optional[str] = "en") -> List[str]:
    """
    Tokenize a sentence into a list of word tokens. Punctuation is merged with
    the token immediately before it when possible.
    """
    if not sentence:
        return []

    nlp = _get_spacy_pipeline(lang or "en")
    doc = nlp(sentence)

    tokens: List[str] = []
    for tok in doc:
        text = tok.text
        if tok.is_punct and tokens:
            tokens[-1] = f"{tokens[-1]}{text}"
        else:
            tokens.append(text)

    return tokens
