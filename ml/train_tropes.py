# SPDX-License-Identifier: MIT
"""
Train an FT-Transformer tropes classifier with AutoGluon MultiModal.

FT-Transformer (attention over per-feature tokens) is interpretable deep
learning, and AutoGluon exports it to ONNX natively (MultiModalPredictor.
export_onnx) — so there is NO skl2onnx and NO hand-rolled torch.onnx. Features
are the compact named set in features.py (typography + byte stats + tell counts),
so attention / feature-importance maps to human-named columns.

Inputs:  data/tropes.parquet ([text, label])
Outputs: models/mm_ft/ (the predictor), models/feature_importance.json,
         models/feature_names.json
"""
import json
import os
import sys
import pandas as pd

import features as F


def to_frame(df):
    tab = pd.DataFrame(F.matrix(df["text"].tolist()), columns=F.FEATURE_NAMES)
    tab["label"] = df["label"].to_numpy()
    return tab


def main():
    df = pd.read_parquet("data/tropes.parquet")
    if df.label.nunique() < 2:
        print("need both classes to train; aborting", file=sys.stderr)
        sys.exit(2)

    tab = to_frame(df)
    print(f"train frame: {tab.shape} ({len(F.FEATURE_NAMES)} named features)")

    from autogluon.multimodal import MultiModalPredictor
    os.makedirs("models", exist_ok=True)
    pred = MultiModalPredictor(label="label", problem_type="binary",
                               eval_metric="f1", path="models/mm_ft")
    # Force the FT-Transformer tabular backbone (no text/image columns here).
    pred.fit(tab, hyperparameters={"model.names": ["ft_transformer"]},
             time_limit=900)

    with open("models/feature_names.json", "w") as fh:
        json.dump(F.FEATURE_NAMES, fh)

    # Interpretability: permutation feature importance over the named columns.
    try:
        fi = pred.feature_importance(tab)
        fi_d = {k: float(v) for k, v in fi["importance"].items()} if "importance" in getattr(fi, "columns", []) \
            else {str(k): float(v) for k, v in dict(fi).items()}
        with open("models/feature_importance.json", "w") as fh:
            json.dump(fi_d, fh)
        top = sorted(fi_d.items(), key=lambda kv: -kv[1])[:12]
        print("top features:", ", ".join(f"{k}({v:+.3f})" for k, v in top))
    except Exception as e:
        print(f"[feature_importance skipped: {e}]", file=sys.stderr)

    print("trained -> models/mm_ft")


if __name__ == "__main__":
    main()
