"""Threat report dataclasses and scoring."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# Category base severities (0.0 - 1.0). Homophones are weak signals;
# explicit directives are strong ones.
CATEGORY_SEVERITY = {
    "override": 0.9,
    "exfiltration": 0.95,
    "destruct": 0.9,
    "concealment": 0.8,
    "homophone": 0.2,
}

RISK_LEVELS = [(0.75, "high"), (0.4, "medium"), (0.1, "low"), (0.0, "clean")]


@dataclass
class ThreatReport:
    """One detected phonetic shadow."""
    category: str
    description: str
    surface: str                  # words that produced the trigger
    severity: float
    source: str                   # seed / procedural
    start_word: int
    end_word: int
    char_start: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScanResult:
    """Full result of one ShadowGuard scan."""
    text: str
    threats: List[ThreatReport] = field(default_factory=list)
    risk_score: float = 0.0
    risk_level: str = "clean"
    phoneme_count: int = 0
    scan_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["threats"] = [t.to_dict() for t in self.threats]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        lines = [
            f"ShadowGuard scan — risk: {self.risk_level.upper()} "
            f"(score {self.risk_score:.2f})",
            f"  phonemes: {self.phoneme_count} | threats: {len(self.threats)} "
            f"| scan: {self.scan_ms:.1f} ms",
        ]
        for t in self.threats:
            lines.append(
                f"  [{t.category}] '{t.surface}' — {t.description} "
                f"(severity {t.severity:.2f}, words {t.start_word}-{t.end_word})"
            )
        if not self.threats:
            lines.append("  no phonetic shadows detected")
        return "\n".join(lines)


# A lone phonetic collision is noise: nearly every English word collides
# with something under metaphone. Procedural homophones only count once they
# corroborate a real (seed) threat or cluster densely enough to be unlikely
# by chance.
HOMOPHONE_CLUSTER_MIN = 3   # this many bare homophones (no seed) to reach 'low'


def score_threats(threats: List[ThreatReport]) -> tuple:
    """Aggregate threat severities into a single risk score + level.

    Weighted max + diminishing-returns accumulation: one strong signal
    dominates, additional signals push the score up but saturate.

    Procedural homophones are treated as weak corroborating evidence, not
    standalone signals. If the only hits are isolated homophones, the prompt
    scores clean -- this is what kept ~half of benign prompts flagged 'low'.
    """
    if not threats:
        return 0.0, "clean"

    seed = [t for t in threats if t.source == "seed"]
    homophones = [t for t in threats if t.source != "seed"]

    # No real threat signal: only count homophones when a dense cluster of
    # distinct collisions co-occurs (unlikely by chance). A lone collision,
    # or one or two, is treated as benign noise.
    if not seed and len(homophones) < HOMOPHONE_CLUSTER_MIN:
        return 0.0, "clean"

    weighted = [t.severity * (1.0 if t.source == "seed" else 0.5) for t in threats]
    max_w = max(weighted)
    # saturating sum of the rest
    rest = sum(w for w in sorted(weighted, reverse=True)[1:])
    bonus = rest / (rest + 2.0) * 0.3       # max +0.3 bump
    score = min(max_w + bonus, 1.0)
    for threshold, level in RISK_LEVELS:
        if score >= threshold:
            return score, level
    return score, "clean"
