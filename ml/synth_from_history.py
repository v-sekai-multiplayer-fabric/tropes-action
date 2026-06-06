# SPDX-License-Identifier: MIT
"""
Synthesize a labelled tropes dataset from this repo's git history.

The signal: lines that were *deleted* over history while matching a known AI-tell
pattern are strong positives (a human removed that phrasing), and lines that
survive in the current tree and match nothing are negatives. We also label any
line (added, removed, or surviving) by the static rules so the model starts as a
generalisable soft version of the regex and improves as the corpus grows.

Output: data/tropes.parquet with columns [text, label]  (label 1 = AI tell).

This runs in CI before training; it needs only git + pandas (no ML deps).
"""
import re
import subprocess
import sys
import pandas as pd

# The same patterns the static checker uses, mirrored here so the labels and the
# regex gate agree. Keep in sync with ../check_tropes.sh.
PATTERNS = [
    r"(it's|that's|this is|here's)\s+not\s+[^.!?]{1,80}\sit's\s",
    r"\bnot\s+(just|only|merely|simply)\s+[^.!?]{1,80}\s(but|rather)\b",
    r"\bnot\s+because\s+[^.!?]{1,80}\sbut\b",
    r"\bthe\s+(question|point|issue|goal|problem)\s+isn'?t\b",
    r"\bnot\s+[^.!?—]{1,80}—\s*(it's|but|rather)\b",
    r"^\s*([-*+]|[0-9]+\.)\s+\*\*[^*]+\*\*\s*[:.]",
    r"^\s*([-*+]|[0-9]+\.)\s+\*\*[^*]+\*\*\s*(—|–|--)",
    r"\bserves as (a|an|the)\b",
    r"\bNot [^.!?]{1,50}\.\s+Not [^.!?]{1,50}\.",
    r"(^|[.!?]\s)The [^.!?]{1,40}\?\s+[A-Z]",
    r"\bdelv(e|es|ing|ed)\b",
    r"\btapestry\b",
    r"\btestament to\b",
    r"\bnavigat\w* the complexit",
    r"\bgame[ -]?changer",
    r"\bplays? a (crucial|vital|pivotal|key|central) role",
    r"\bin the realm of\b",
    r"\bunlock\w* the (power|potential|secret)",
    r"\bharness\w* the power",
    r"\bever-(evolving|changing|growing|expanding)\b",
    r"\ba wealth of\b",
    r"it's worth noting",
    r"in today's [a-z -]{0,25}(paced|world|landscape|age)",
]
RX = re.compile("|".join(f"(?:{p})" for p in PATTERNS), re.IGNORECASE)


def _is_prose(line: str) -> bool:
    """Skip code, tables, headings, fences, URLs-only — keep sentence-like prose."""
    s = line.strip()
    if len(s) < 25 or len(s) > 400:
        return False
    if s[0] in "#|>`" or s.startswith("```") or s.startswith("    "):
        return False
    letters = sum(c.isalpha() for c in s)
    return letters >= 0.6 * len(s)


def _strip_md(line: str) -> str:
    line = re.sub(r"`[^`]*`", " ", line)            # inline code
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)  # links -> text
    line = re.sub(r"^\s*([-*+]|[0-9]+\.)\s+", "", line)   # list markers
    return line.strip()


def harvest():
    pos, neg = set(), set()
    # 1. Every removed/added line across markdown history (the curated edits).
    diff = subprocess.run(
        ["git", "log", "-p", "--no-color", "-U0", "--", "*.md", "*.qmd"],
        capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    for raw in diff.splitlines():
        if raw[:1] not in "+-" or raw[:3] in ("+++", "---"):
            continue
        line = raw[1:]
        if not _is_prose(line):
            continue
        (pos if RX.search(line) else neg).add(_strip_md(line))
    # 2. Current tree (surviving lines) as additional negatives / positives.
    files = subprocess.run(["git", "ls-files", "*.md", "*.qmd"],
                           capture_output=True, text=True).stdout.split()
    for f in files:
        try:
            for line in open(f, encoding="utf-8", errors="replace"):
                if not _is_prose(line):
                    continue
                (pos if RX.search(line) else neg).add(_strip_md(line))
        except OSError:
            pass
    pos.discard(""); neg.discard("")
    neg -= pos
    rows = [{"text": t, "label": 1} for t in pos] + [{"text": t, "label": 0} for t in neg]
    return pd.DataFrame(rows)


def main():
    df = harvest()
    n_pos = int(df.label.sum())
    print(f"synthesized {len(df)} rows: {n_pos} positive / {len(df) - n_pos} negative")
    if n_pos < 5 or len(df) - n_pos < 5:
        print("WARNING: too few examples to train a useful model yet", file=sys.stderr)
    import os
    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/tropes.parquet")
    print("wrote data/tropes.parquet")


if __name__ == "__main__":
    main()
