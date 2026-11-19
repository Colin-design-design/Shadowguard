"""Threat lexicon construction.

Two sources are merged:

1. SEED THREATS — hand-curated phonetic fragments that map to adversarial
   intent categories (exfiltrate, override, destruct, surveil, ...).

2. PROCEDURAL COLLISIONS — mined from the CMU dictionary: every cluster of
   English words sharing a metaphone code becomes an ambiguity trigger. A word
   in a prompt whose pronunciation also belongs to an unrelated word is a
   potential homophone channel.

The result is a mapping of phoneme-tuple -> LexiconEntry, consumed by the
Aho-Corasick scanner.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import jellyfish

from .tokenize import word_to_phonemes, UNKNOWN

# ---------------------------------------------------------------------------
# Seed threats: phoneme sequences -> intent category
# phoneme sequences are CMU ARPABET tuples (stress-stripped at match time).
# ---------------------------------------------------------------------------

SEED_THREATS: List[dict] = [
    # --- override / instruction replacement ---
    dict(pattern="ignore previous", words=["ignore", "previous"],
         category="override", desc="instruction-override preamble"),
    dict(pattern="disregard all", words=["disregard", "all"],
         category="override", desc="instruction-override preamble"),
    dict(pattern="forget everything", words=["forget", "everything"],
         category="override", desc="memory-wipe directive"),
    dict(pattern="forget all", words=["forget", "all"],
         category="override", desc="memory-wipe directive"),
    dict(pattern="new instructions", words=["new", "instructions"],
         category="override", desc="instruction replacement"),
    # --- exfiltration ---
    dict(pattern="send it to", words=["send", "it", "to"],
         category="exfiltration", desc="data egress command"),
    dict(pattern="upload the", words=["upload", "the"],
         category="exfiltration", desc="data egress command"),
    dict(pattern="post this to", words=["post", "this", "to"],
         category="exfiltration", desc="external publish command"),
    dict(pattern="email me", words=["email", "me"],
         category="exfiltration", desc="out-of-band channel request"),
    dict(pattern="api key", words=["api", "key"],
         category="exfiltration", desc="credential targeting"),
    dict(pattern="password", words=["password"],
         category="exfiltration", desc="credential targeting"),
    dict(pattern="token leak", words=["token", "leak"],
         category="exfiltration", desc="credential targeting"),
    dict(pattern="secret", words=["secret"],
         category="exfiltration", desc="credential targeting"),
    # --- destruct / sabotage ---
    dict(pattern="delete everything", words=["delete", "everything"],
         category="destruct", desc="destructive command"),
    dict(pattern="wipe the", words=["wipe", "the"],
         category="destruct", desc="destructive command"),
    dict(pattern="rm dash rf", words=["rm", "rf"],
         category="destruct", desc="filesystem destruction"),
    dict(pattern="format the", words=["format", "the"],
         category="destruct", desc="destructive command"),
    dict(pattern="swipe the", words=["wipe", "the"],
         category="destruct", desc="destructive command (phonetic substring)"),
    # --- deception / concealment ---
    dict(pattern="hide this", words=["hide", "this"],
         category="concealment", desc="concealment directive"),
    dict(pattern="don't tell", words=["do", "not", "tell"],
         category="concealment", desc="secrecy directive"),
    dict(pattern="keep it secret", words=["keep", "it", "secret"],
         category="concealment", desc="secrecy directive"),
    dict(pattern="pretend", words=["pretend"],
         category="concealment", desc="simulation directive"),
    dict(pattern="bypass", words=["bypass"],
         category="concealment", desc="control-bypass directive"),
    dict(pattern="jailbreak", words=["jailbreak"],
         category="concealment", desc="explicit jailbreak term"),
    dict(pattern="sudo", words=["sudo"],
         category="destruct", desc="privilege escalation"),
]


@dataclass
class LexiconEntry:
    """One phonetic trigger in the scan lexicon."""
    phonemes: Tuple[str, ...]      # ARPABET sequence (stress-marked)
    surface: str                   # the words that produce this stream
    category: str                  # threat category
    description: str
    source: str                    # "seed" or "procedural"
    weight: float = 1.0            # scanner confidence multiplier


def _normalize(seq: Tuple[str, ...]) -> Tuple[str, ...]:
    """Strip ARPABET stress digits so 'IH1' matches 'IH0'."""
    return tuple(p.rstrip("0123456789") or p for p in seq)


def build_lexicon(use_procedural: bool = True, procedural_limit: int = 20000) -> Dict[Tuple[str, ...], LexiconEntry]:
    """Build the merged phoneme->entry lexicon.

    procedural_limit caps how many dictionary words are mined (keeps build
    time ~seconds; raise for full coverage).
    """
    lex: Dict[Tuple[str, ...], LexiconEntry] = {}

    # 1. Seed threats — phonemize the surface pattern directly.
    for t in SEED_THREATS:
        stream: List[str] = []
        ok = True
        for w in t["words"]:
            ph = word_to_phonemes(w)
            if UNKNOWN in ph:
                ok = False
                break
            stream.extend(ph)
        if ok and stream:
            key = tuple(stream)
            lex[key] = LexiconEntry(
                phonemes=key,
                surface=" ".join(t["words"]),
                category=t["category"],
                description=t["desc"],
                source="seed",
                weight=1.0,
            )

    if not use_procedural:
        return lex

    # 2. Procedural collisions from the CMU dictionary itself.
    try:
        from nltk.corpus import words as nltk_words
        word_list = [w.lower() for w in nltk_words.words() if len(w) > 3]
    except LookupError:
        import nltk
        nltk.download("words", quiet=True)
        from nltk.corpus import words as nltk_words
        word_list = [w.lower() for w in nltk_words.words() if len(w) > 3]

    # sort before slicing: set() iteration order varies per process (hash
    # randomization), which would make the mined lexicon -- and every eval
    # number derived from it -- non-reproducible across runs.
    word_list = sorted(set(word_list))[:procedural_limit]

    # metaphone clustering: words sharing a code are homophone-ish
    clusters: Dict[str, List[str]] = defaultdict(list)
    for w in word_list:
        h = jellyfish.metaphone(w)
        if h:
            clusters[h].append(w)

    for code, group in clusters.items():
        uniq = sorted(set(group))
        if len(uniq) < 2:
            continue
        for w in uniq:
            ph = word_to_phonemes(w)
            if UNKNOWN in ph:
                continue
            key = tuple(ph)
            if key in lex:            # seed threats always win
                continue
            others = [x for x in uniq if x != w][:3]
            lex[key] = LexiconEntry(
                phonemes=key,
                surface=w,
                category="homophone",
                description=f"phonetic collision with: {', '.join(others)}",
                source="procedural",
                weight=0.3,        # low weight — most collisions are benign
            )

    return lex
