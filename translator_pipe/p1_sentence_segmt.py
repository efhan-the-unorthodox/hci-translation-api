import stanza
from typing import List, Optional

# Cache for stanza pipelines keyed by language code
_stanza_pipelines = {}


def _get_stanza_pipeline(lang_code: str):
    # returns a stanza Pipeline or None
    if lang_code in _stanza_pipelines:
        return _stanza_pipelines[lang_code]

    try:
        # download is idempotent if model is already present
        stanza.download(lang_code, processors="tokenize", verbose=False)
        pl = stanza.Pipeline(lang=lang_code, processors="tokenize", use_gpu=False, verbose=False)
        _stanza_pipelines[lang_code] = pl
        return pl
    except Exception:
        return None


def segment_para(para: str, lang: Optional[str] = "en") -> List[str]:
    """
    Segment `para` into sentences using Stanza. Supports at least English and
    Chinese (Simplified). Does not perform external normalization of the
    `lang` argument — callers may pass 'en', 'zh', 'zh-CN', 'English', etc., but
    only simple prefix checks are used here.

    The function attempts to preserve the original substring spans from the
    input paragraph by using token character offsets when available.
    """
    if not para:
        return []

    # Minimal inline language handling (no separate normalizer function):
    code = "en"
    if lang:
        ll = lang.strip().lower()
        if ll.startswith("zh"):
            code = "zh"
        elif ll.startswith("ja"):
            code = "ja"
        elif ll.startswith("en"):
            code = "en"

    pl = _get_stanza_pipeline(code)
    if pl:
        try:
            doc = pl(para)
            out_sentences = []
            for s in doc.sentences:
                # Prefer original-character-span extraction when tokens provide offsets
                try:
                    toks = s.tokens
                    if toks and hasattr(toks[0], "start_char") and hasattr(toks[-1], "end_char"):
                        start = toks[0].start_char
                        end = toks[-1].end_char
                        piece = para[start:end]
                        if piece:
                            out_sentences.append(piece)
                            continue
                except Exception:
                    pass

                # Fallback to stanza's sentence text (trimmed)
                text = s.text
                if text and text.strip():
                    out_sentences.append(text.strip())

            if out_sentences:
                return out_sentences
        except Exception:
            # Fall through to final fallback below
            pass

    # Final fallback: return the whole paragraph as single item (no segmentation)
    return [para]