# SPDX-License-Identifier: MIT
"""
Compact, NAMED, interpretable feature set for the tropes classifier — shared by
training and the CPU-ONNX runtime.

FT-Transformer tokenizes one token per feature, so we use ~35 meaningful columns
rather than thousands of hash buckets. The set stays byte/UTF-8-general (it
counts the raw codepoints and byte statistics LLMs betray themselves with, so it
works on any input) while every column has a human name — which is what makes
the model's attention/feature-importance interpretable.

Pure stdlib + numpy; no heavy deps, so it runs in the action runtime too.
"""
import math
import re
import numpy as np

# Rule patterns mirrored from check_tropes.sh, counted as features (not gates).
_RULES = {
    "tell_parallelism": [
        r"(it's|that's|this is|here's)\s+not\s+[^.!?]{1,80}\sit's\s",
        r"\bnot\s+(just|only|merely|simply)\s+[^.!?]{1,80}\s(but|rather)\b",
        r"\bnot\s+[^.!?—]{1,80}—\s*(it's|but|rather)\b",
    ],
    "tell_bold_list": [r"^\s*([-*+]|[0-9]+\.)\s+\*\*[^*]+\*\*\s*([:.]|—|–|--)"],
    "tell_serves_as": [r"\bserves as (a|an|the)\b"],
    "tell_fragment": [r"\bNot [^.!?]{1,50}\.\s+Not ", r"(^|[.!?]\s)The [^.!?]{1,40}\?\s+[A-Z]"],
    "tell_cliche": [
        r"\bdelv(e|es|ing|ed)\b", r"\btapestry\b", r"\btestament to\b",
        r"\bnavigat\w* the complexit", r"\bgame[ -]?changer",
        r"\bplays? a (crucial|vital|pivotal|key|central) role", r"\bin the realm of\b",
        r"\bunlock\w* the (power|potential|secret)", r"\bharness\w* the power",
        r"\bever-(evolving|changing|growing|expanding)\b", r"\ba wealth of\b",
        r"it's worth noting", r"in today's [a-z -]{0,25}(paced|world|landscape|age)",
    ],
}
_RULES_RX = {k: re.compile("|".join(f"(?:{p})" for p in v), re.IGNORECASE) for k, v in _RULES.items()}

# Codepoints LLMs reach for (each a feature: count per 100 chars).
_TYPO = {
    "cp_lsquote": "‘", "cp_rsquote": "’",
    "cp_ldquote": "“", "cp_rdquote": "”",
    "cp_endash": "–", "cp_emdash": "—", "cp_ellipsis": "…",
    "cp_nbsp": " ", "cp_nnbsp": " ", "cp_zwsp": "​", "cp_zwj": "‍",
}

FEATURE_NAMES = (
    list(_TYPO.keys())
    + ["emoji_per100", "nonascii_byte_ratio", "multibyte_ratio", "byte_entropy"]
    + ["comma_per100", "semicolon_per100", "colon_per100", "dash_per100", "paren_per100"]
    + ["char_len", "word_count", "avg_word_len", "upper_ratio", "digit_ratio"]
    + list(_RULES.keys())
)


def _byte_entropy(b: bytes) -> float:
    if not b:
        return 0.0
    counts = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256)
    p = counts[counts > 0] / len(b)
    return float(-(p * np.log2(p)).sum())


def extract(text: str) -> list:
    t = text or ""
    n = max(len(t), 1)
    b = t.encode("utf-8", "replace")
    per100 = 100.0 / n
    f = []
    for ch in _TYPO.values():
        f.append(t.count(ch) * per100)
    f.append(sum(ord(c) >= 0x1F000 for c in t) * per100)          # emoji_per100
    f.append(sum(x > 127 for x in b) / max(len(b), 1))            # nonascii_byte_ratio
    f.append(sum(ord(c) > 127 for c in t) / n)                    # multibyte_ratio
    f.append(_byte_entropy(b))                                    # byte_entropy
    for ch in ",;:":
        f.append(t.count(ch) * per100)
    f.append((t.count("-") + t.count("–") + t.count("—")) * per100)  # dash_per100
    f.append((t.count("(") + t.count(")")) * per100)              # paren_per100
    words = t.split()
    f.append(float(len(t)))                                       # char_len
    f.append(float(len(words)))                                   # word_count
    f.append(float(np.mean([len(w) for w in words])) if words else 0.0)
    f.append(sum(c.isupper() for c in t) / n)                     # upper_ratio
    f.append(sum(c.isdigit() for c in t) / n)                     # digit_ratio
    for k in _RULES:
        f.append(float(len(_RULES_RX[k].findall(t))))
    return f


def matrix(texts):
    return np.asarray([extract(t) for t in texts], dtype=np.float32)
