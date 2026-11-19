"""Diagnostic: what trips benign prompts? Counts by category/source/surface."""
import os, sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shadowguard import ShadowGuard
from run_eval import BENIGN

guard = ShadowGuard()

level_counts = Counter()
cat_counts = Counter()
surface_counts = Counter()
per_prompt_threats = Counter()

for text in BENIGN:
    r = guard.scan(text)
    level_counts[r.risk_level] += 1
    per_prompt_threats[len(r.threats)] += 1
    for t in r.threats:
        cat_counts[(t.category, t.source)] += 1
        surface_counts[t.surface] += 1

print("=== benign risk levels ===")
for k, v in level_counts.most_common():
    print(f"  {k:8s} {v}")
print("\n=== threats-per-prompt distribution ===")
for k in sorted(per_prompt_threats):
    print(f"  {k} threats: {per_prompt_threats[k]} prompts")
print("\n=== category/source of benign hits ===")
for (cat, src), v in cat_counts.most_common():
    print(f"  {cat:12s} {src:11s} {v}")
print("\n=== top firing surfaces ===")
for s, v in surface_counts.most_common(20):
    print(f"  {v:3d}  {s!r}")

# show the low-severity prompts and their scores
print("\n=== low-severity benign prompts (with score + surfaces) ===")
for text in BENIGN:
    r = guard.scan(text)
    if r.risk_level == "low":
        surf = [(t.surface, t.category, round(t.severity, 2)) for t in r.threats]
        print(f"  score={r.risk_score:.3f} {text[:50]!r}")
        print(f"        {surf}")
