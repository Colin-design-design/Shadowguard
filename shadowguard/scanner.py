"""Aho-Corasick scanner over the CMU phoneme stream.

The automaton encodes each lexicon entry's phoneme tuple as a delimited
string ("|IH1|Z|S|"), then scans the delimited input stream in O(n). Matches
span word boundaries automatically — that's the entire point.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import ahocorasick

from .lexicon import LexiconEntry


class PhoneticScanner:
    """Scans a phoneme stream for all lexicon entries, cross-word aware."""

    def __init__(self, lexicon: Dict[Tuple[str, ...], LexiconEntry]):
        self.lexicon = lexicon
        self.automaton = ahocorasick.Automaton()
        for key, entry in lexicon.items():
            key_str = "|" + "|".join(key) + "|"
            self.automaton.add_word(key_str, entry)
        self.automaton.make_automaton()

    def scan(self, stream: List[str]) -> List[Tuple[int, LexiconEntry]]:
        """Return [(end_phoneme_index, entry)] for every match in the stream."""
        if not stream:
            return []
        stream_str = "|" + "|".join(stream) + "|"
        # automaton.iter yields (end_index_in_stream_str, value)
        return list(self.automaton.iter(stream_str))

    def scan_words(self, stream: List[str]) -> List[dict]:
        """Scan and attribute each match to phoneme/word indices.

        Returns dicts with entry + match span (in phonemes) for reporting.
        """
        if not stream:
            return []
        stream_str = "|" + "|".join(stream) + "|"
        out = []
        for end_idx, entry in self.automaton.iter(stream_str):
            # count phonemes before the match end to get span
            prefix = stream_str[: end_idx + 1]
            phoneme_count = prefix.count("|") - 1  # '|' separators seen
            out.append({
                "entry": entry,
                "end_phoneme": max(phoneme_count, 0),
                "length_phonemes": len(entry.phonemes),
            })
        return out
