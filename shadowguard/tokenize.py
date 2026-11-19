"""Tokenizer: text -> continuous CMU phoneme stream.

This is the core primitive of phonetic shadow detection. Words are converted
to CMU pronunciations and concatenated into ONE stream, so cross-word
fragments ("is snowy" -> "is no we") become detectable.
"""
from __future__ import annotations

import functools
import re
from typing import List

try:
    from nltk.corpus import cmudict
except LookupError:  # pragma: no cover - handled in ensure_nltk
    from nltk import download as _dl
    _dl("cmudict", quiet=True)
    from nltk.corpus import cmudict

UNKNOWN = "[UNKNOWN]"

_TOKEN_RE = re.compile(r"[a-z']+")
_APOS_RE = re.compile(r"^'+|'+$")


def ensure_nltk() -> None:
    """Download cmudict if missing (no-op after first run)."""
    try:
        cmudict.dict()
    except LookupError:
        import nltk
        nltk.download("cmudict", quiet=True)


@functools.lru_cache(maxsize=1)
def _dict() -> dict:
    ensure_nltk()
    return cmudict.dict()


def word_to_phonemes(word: str) -> List[str]:
    """Primary CMU pronunciation for a word, or [UNKNOWN]."""
    d = _dict()
    key = word.lower()
    if key in d:
        return list(d[key][0])
    return [UNKNOWN]


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, return word tokens."""
    words = _TOKEN_RE.findall(text.lower())
    return [_APOS_RE.sub("", w) for w in words if _APOS_RE.sub("", w)]


def to_phoneme_stream(text: str) -> List[str]:
    """Flatten text into one continuous phoneme stream (cross-word boundary aware)."""
    stream: List[str] = []
    for w in tokenize(text):
        stream.extend(word_to_phonemes(w))
    return stream


def stream_to_positions(text: str) -> List[tuple]:
    """Return [(phoneme, word_index, char_start)] for report attribution.

    char_start is approximate (word-level granularity) — enough to point the
    user at which part of the prompt triggered a shadow.
    """
    d = _dict()
    result = []
    words = tokenize(text)
    # locate word offsets in original text (case-insensitive scan)
    lowered = text.lower()
    search_from = 0
    offsets = []
    for w in words:
        idx = lowered.find(w, search_from)
        if idx == -1:
            idx = search_from
        offsets.append(idx)
        search_from = idx + len(w)
    for wi, w in enumerate(words):
        key = w.lower()
        ph = list(d[key][0]) if key in d else [UNKNOWN]
        for p in ph:
            result.append((p, wi, offsets[wi]))
    return result
