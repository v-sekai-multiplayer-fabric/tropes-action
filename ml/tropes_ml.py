# SPDX-License-Identifier: MIT
"""
Torch-free ONNX tropes scorer: onnxruntime + numpy only (mirrors
jobs-lazy-onboarding/embed.py). Scores Markdown prose for AI-tell likelihood on
the CPU, and — because the deployed model is linear over n-grams — attributes
each flag to the specific phrases that drove it (explain.json).

This is the ADVISORY layer: it prints warnings, it does not fail the build (the
static check_tropes.sh stays the deterministic gate). The model is fetched
lazily: pass --model-dir pointing at a downloaded onnx_tropes/, or the CI/action
places it next to this file.

Usage:
  python tropes_ml.py --model-dir onnx_tropes [--threshold 0.6] [files...]
"""
import argparse
import json
import os
import re
import subprocess
import sys

# Provider preference: cross-vendor GPU first, CPU always available — same order
# as jobs-lazy-onboarding so the action accelerates anywhere but never requires it.
_PROVIDER_PREF = ["CoreMLExecutionProvider", "DmlExecutionProvider",
                  "VulkanExecutionProvider", "WebGpuExecutionProvider",
                  "ROCMExecutionProvider", "CUDAExecutionProvider",
                  "CPUExecutionProvider"]


class TropesModel:
    def __init__(self, model_dir):
        import numpy as np
        import onnxruntime as ort
        self.np = np
        avail = set(ort.get_available_providers())
        providers = [p for p in _PROVIDER_PREF if p in avail] or ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(os.path.join(model_dir, "model.onnx"),
                                         providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        with open(os.path.join(model_dir, "explain.json"), encoding="utf-8") as fh:
            ex = json.load(fh)
        # Positive-weight n-grams only, longest first, for phrase attribution.
        self.weights = {k: v for k, v in ex["weights"].items() if v > 0}

    def score(self, texts):
        arr = self.np.array([[t] for t in texts], dtype=object)
        res = self.sess.run(None, {self.in_name: arr})
        proba = next(r for r in res if getattr(r, "ndim", 0) == 2 and r.shape[1] == 2)
        return proba[:, 1]

    def attribute(self, text, k=4):
        low = text.lower()
        hits = [(ng, w) for ng, w in self.weights.items() if ng.strip() and ng in low]
        hits.sort(key=lambda kv: -kv[1])
        return [ng.strip() for ng, _ in hits[:k]]


def _prose_lines(path):
    out = []
    for i, line in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
        s = line.strip()
        if len(s) >= 25 and s[:1] not in "#|>`" and not s.startswith("```"):
            out.append((i, s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="onnx_tropes")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()

    if not os.path.exists(os.path.join(a.model_dir, "model.onnx")):
        print(f"[tropes-ml] no model at {a.model_dir}; skipping advisory pass")
        return 0

    files = a.files or subprocess.run(
        ["git", "ls-files", "*.md", "*.qmd"], capture_output=True, text=True
    ).stdout.split()

    model = TropesModel(a.model_dir)
    flagged = 0
    for f in files:
        rows = _prose_lines(f)
        if not rows:
            continue
        scores = model.score([s for _, s in rows])
        for (lineno, s), p in zip(rows, scores):
            if p >= a.threshold:
                why = model.attribute(s)
                because = f"  (phrases: {', '.join(why)})" if why else ""
                print(f"tropes-ml: {f}:{lineno}: p(AI-tell)={p:.2f}{because}")
                flagged += 1
    print(f"[tropes-ml] advisory: {flagged} line(s) above {a.threshold:.2f}")
    return 0  # advisory: never fails the build


if __name__ == "__main__":
    sys.exit(main())
