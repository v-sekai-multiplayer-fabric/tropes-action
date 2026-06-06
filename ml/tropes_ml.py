# SPDX-License-Identifier: MIT
"""
CPU-ONNX tropes scorer for the FT-Transformer model. Featurizes each prose line
into the named feature set (features.py), runs the AutoGluon-exported ONNX graph
on the CPU, and explains a flag via the globally most-important features that are
active on that line (feature_importance.json) plus any typographic tells.

Advisory only — prints warnings, never fails the build. Model fetched lazily by
the action; pass --model-dir to a folder with model.onnx + feature_names.json
(+ optional feature_importance.json, io.json).

Usage: python tropes_ml.py --model-dir onnx_tropes [--threshold 0.6] [files...]
"""
import argparse
import json
import os
import subprocess
import sys
import numpy as np

import features as F

_PROVIDER_PREF = ["CoreMLExecutionProvider", "DmlExecutionProvider",
                  "VulkanExecutionProvider", "WebGpuExecutionProvider",
                  "ROCMExecutionProvider", "CUDAExecutionProvider",
                  "CPUExecutionProvider"]
_TYPO_FEATURES = [n for n in F.FEATURE_NAMES if n.startswith("cp_") or n == "emoji_per100"]


class TropesModel:
    def __init__(self, model_dir):
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
        providers = [p for p in _PROVIDER_PREF if p in avail] or ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(os.path.join(model_dir, "model.onnx"),
                                         providers=providers)
        self.inputs = self.sess.get_inputs()
        imp_path = os.path.join(model_dir, "feature_importance.json")
        self.importance = {}
        if os.path.exists(imp_path):
            with open(imp_path) as fh:
                self.importance = json.load(fh)

    def _feed(self, X):
        # FT-Transformer ONNX with all-numeric input is typically a single float
        # tensor [batch, n_features]; introspect to stay robust to the signature.
        floats = [i for i in self.inputs if "float" in str(i.type)]
        if len(self.inputs) == 1 and floats:
            return {self.inputs[0].name: X}
        # Fallback: map any single float input to the feature matrix.
        if floats:
            return {floats[0].name: X}
        raise RuntimeError(f"unexpected ONNX inputs: {[i.name for i in self.inputs]}")

    def score(self, texts):
        X = F.matrix(texts)
        res = self.sess.run(None, self._feed(X))
        proba = next((r for r in res if getattr(r, "ndim", 0) == 2 and r.shape[1] == 2), None)
        if proba is None:
            proba = res[0]
        return proba[:, 1]

    def why(self, text):
        vec = dict(zip(F.FEATURE_NAMES, F.extract(text)))
        active = {k: v for k, v in vec.items() if v}
        # Rank active features by global importance (fallback: by value).
        ranked = sorted(active, key=lambda k: -(self.importance.get(k, 0.0) or active[k]))
        bits = [f"{k}={active[k]:.2g}" for k in ranked[:4]]
        typo = [k for k in _TYPO_FEATURES if vec.get(k)]
        if typo:
            bits.append("typo:" + ",".join(t.replace("cp_", "") for t in typo))
        return "  (" + "; ".join(bits) + ")" if bits else ""


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
                print(f"tropes-ml: {f}:{lineno}: p(AI-tell)={p:.2f}{model.why(s)}")
                flagged += 1
    print(f"[tropes-ml] advisory: {flagged} line(s) above {a.threshold:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
