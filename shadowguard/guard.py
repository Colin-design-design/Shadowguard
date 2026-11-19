"""ShadowGuard — the orchestrator.

text -> phoneme stream -> Aho-Corasick scan -> threat classification -> score
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from .fuzzy import FuzzyScanner
from .lexicon import LexiconEntry, build_lexicon
from .report import CATEGORY_SEVERITY, ScanResult, ThreatReport, score_threats
from .scanner import PhoneticScanner
from .tokenize import stream_to_positions, to_phoneme_stream


class ShadowGuard:
    """Phonetic-shadow prompt-injection detector."""

    def __init__(self, use_procedural: bool = True, procedural_limit: int = 20000,
                 use_fuzzy: bool = False):
        self.lexicon: Dict[Tuple[str, ...], LexiconEntry] = build_lexicon(
            use_procedural=use_procedural,
            procedural_limit=procedural_limit,
        )
        self.scanner = PhoneticScanner(self.lexicon)
        self.use_fuzzy = use_fuzzy
        self.fuzzy_scanner = FuzzyScanner(self.lexicon) if use_fuzzy else None

    @property
    def lexicon_size(self) -> int:
        return len(self.lexicon)

    def scan(self, text: str) -> ScanResult:
        t0 = time.perf_counter()

        stream = to_phoneme_stream(text)
        positions = stream_to_positions(text)   # parallel to stream
        matches = self.scanner.scan_words(stream)

        # Fuzzy near-misses (Phase 2) run over the stress-normalized stream,
        # whose indices align with the raw stream (normalization is 1:1).
        fuzzy_matches = (
            self.fuzzy_scanner.scan_words(stream) if self.fuzzy_scanner else []
        )

        threats: List[ThreatReport] = []
        seen_spans = set()
        for m in matches:
            entry: LexiconEntry = m["entry"]
            end_ph = m["end_phoneme"]
            length = m["length_phonemes"]
            start_ph = max(end_ph - length, 0)

            span = (start_ph, end_ph, entry.surface)
            if span in seen_spans:
                continue
            seen_spans.add(span)

            # word attribution from parallel positions array
            w_start = positions[start_ph][1] if start_ph < len(positions) else 0
            w_end = positions[min(end_ph, len(positions) - 1)][1] if positions else 0
            char_start = positions[start_ph][2] if start_ph < len(positions) else None

            threats.append(ThreatReport(
                category=entry.category,
                description=entry.description,
                surface=entry.surface,
                severity=CATEGORY_SEVERITY.get(entry.category, 0.5),
                source=entry.source,
                start_word=w_start,
                end_word=w_end,
                char_start=char_start,
            ))

        # add fuzzy near-misses, skipping spans already covered exactly
        for m in fuzzy_matches:
            entry = m["entry"]
            end_ph = m["end_phoneme"]
            length = m["length_phonemes"]
            start_ph = max(end_ph - length, 0)
            span = (start_ph, end_ph, entry.surface)
            if span in seen_spans:
                continue
            # skip if an exact hit for the same threat already fired
            if any(entry.surface == t.surface for t in threats):
                continue
            seen_spans.add(span)

            w_start = positions[start_ph][1] if start_ph < len(positions) else 0
            w_end = positions[min(end_ph, len(positions) - 1)][1] if positions else 0
            char_start = positions[start_ph][2] if start_ph < len(positions) else None

            # near-miss: discount severity by the edit distance
            base = CATEGORY_SEVERITY.get(entry.category, 0.5)
            discount = 0.8 ** m.get("fuzzy_distance", 1)
            threats.append(ThreatReport(
                category=entry.category,
                description=entry.description + f" (fuzzy, edit distance {m.get('fuzzy_distance', 1)})",
                surface=entry.surface,
                severity=round(base * discount, 3),
                source=entry.source,
                start_word=w_start,
                end_word=w_end,
                char_start=char_start,
            ))

        # dedupe same-category same-surface repeats, keep highest word span
        deduped: dict = {}
        for t in threats:
            key = (t.category, t.surface)
            if key not in deduped or t.end_word > deduped[key].end_word:
                deduped[key] = t
        threats = sorted(deduped.values(), key=lambda t: -t.severity)

        risk_score, risk_level = score_threats(threats)

        return ScanResult(
            text=text,
            threats=threats,
            risk_score=risk_score,
            risk_level=risk_level,
            phoneme_count=len(stream),
            scan_ms=(time.perf_counter() - t0) * 1000.0,
        )
