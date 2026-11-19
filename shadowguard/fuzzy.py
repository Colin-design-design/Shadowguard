"""Phase-2 fuzzy phoneme matching.

Exact Aho-Corasick catches a threat only when its phoneme sequence appears
verbatim in the stream. Real phonetic-shadow attacks drift by an inserted,
dropped, or swapped phoneme across word boundaries:

    "is snowy"  -> IH Z S N OW IY
    "is no we"  -> IH Z   N OW W IY   (1 deletion + 1 insertion away)

This module slides a stress-normalized window over the phoneme stream and
flags windows within a small edit distance of a *seed* threat sequence.

Scope: seed threats only. Fuzzy-matching the ~1,100 procedural homophones
would re-explode the false-positive rate; fuzzy-matching ~25 curated threat
phrases catches obfuscation at bounded cost.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .lexicon import LexiconEntry, _normalize

# minimum seed length (in phonemes) eligible for fuzzy matching: short
# sequences match too much by chance.
MIN_FUZZY_PHONEMES = 4
# allowed edit distance as a fraction of the pattern length (rounded),
# floored at 1. 0.25 -> a 1-phoneme drift per 4 phonemes.
FUZZY_RATIO = 0.25


def _bounded_levenshtein(a: Tuple[str, ...], b: Tuple[str, ...], max_d: int) -> int:
    """Levenshtein distance over phoneme tuples, early-exit above max_d.

    Returns max_d + 1 if the distance provably exceeds max_d.
    """
    la, lb = len(a), len(b)
    if abs(la - lb) > max_d:
        return max_d + 1
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_min = cur[0]
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if cur[j] < row_min:
                row_min = cur[j]
        if row_min > max_d:
            return max_d + 1
        prev = cur
    return prev[lb]


class FuzzyScanner:
    """Edit-distance-tolerant matcher for seed threats over a phoneme stream."""

    def __init__(self, lexicon: Dict[Tuple[str, ...], LexiconEntry]):
        # keep only seed entries, pre-normalize stress, skip short ones
        self.patterns: List[Tuple[Tuple[str, ...], LexiconEntry, int]] = []
        for entry in lexicon.values():
            if entry.source != "seed":
                continue
            norm = _normalize(entry.phonemes)
            if len(norm) < MIN_FUZZY_PHONEMES:
                continue
            max_d = max(1, round(len(norm) * FUZZY_RATIO))
            self.patterns.append((norm, entry, max_d))

    def scan_words(self, stream: List[str]) -> List[dict]:
        """Return fuzzy matches as guard-compatible dicts.

        Only near-misses are reported (edit distance >= 1); exact hits are
        already covered by the Aho-Corasick scanner. Each match records the
        best (lowest-distance) window per pattern to avoid span spam.
        """
        if not stream:
            return []
        norm_stream = list(_normalize(tuple(stream)))
        n = len(norm_stream)
        out: List[dict] = []

        for pat, entry, max_d in self.patterns:
            lp = len(pat)
            best = None  # (distance, start, end)
            # window lengths within +/- max_d of the pattern length
            for wlen in range(max(1, lp - max_d), lp + max_d + 1):
                if wlen > n:
                    break
                for start in range(0, n - wlen + 1):
                    window = tuple(norm_stream[start:start + wlen])
                    d = _bounded_levenshtein(pat, window, max_d)
                    if 1 <= d <= max_d and (best is None or d < best[0]):
                        best = (d, start, start + wlen)
            if best is not None:
                _, start, end = best
                out.append({
                    "entry": entry,
                    "end_phoneme": end,
                    "length_phonemes": end - start,
                    "fuzzy_distance": best[0],
                })
        return out
