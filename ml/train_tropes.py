# SPDX-License-Identifier: MIT
"""
Train an EXPLAINABLE tropes classifier and emit an ONNX-exportable pipeline.

Two things happen here:

  1. AutoGluon Tabular with `presets="interpretable"` fits rule-based models
     (imodels) on the synthesized data and prints the learned rules — this is the
     human-readable "why" we keep in the run log / EXPLAIN.md. AutoGluon also
     gives a leaderboard so we know the signal is real before shipping.

  2. The DEPLOYED model is a small, fully ONNX-exportable and inherently
     explainable sklearn pipeline: word+char n-gram TF-IDF -> LogisticRegression.
     Its coefficients map every n-gram to a signed weight, so at inference we can
     point at the exact phrases that drove a flag (see tropes_ml.py).

Inputs:  data/tropes.parquet  ([text, label])
Outputs: models/pipeline.joblib, models/explain.json (vocab -> weight, intercept)
"""
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd


def train_sklearn(df):
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    feats = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 3),
                                 min_df=2, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=3, sublinear_tf=True, lowercase=True)),
    ])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)
    pipe = Pipeline([("feats", feats), ("clf", clf)])
    pipe.fit(df["text"].tolist(), df["label"].values)
    return pipe


def dump_explanations(pipe, path):
    """Flatten the linear coefficients to a {feature: weight} table so the
    runtime can attribute a score to specific n-grams without sklearn."""
    feats = pipe.named_steps["feats"]
    clf = pipe.named_steps["clf"]
    names = feats.get_feature_names_out()
    coef = clf.coef_[0]
    # Keep only the meaningfully-weighted features to keep the file small.
    keep = {n: round(float(w), 5) for n, w in zip(names, coef) if abs(w) > 0.05}
    payload = {"intercept": float(clf.intercept_[0]), "weights": keep}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    top = sorted(keep.items(), key=lambda kv: -kv[1])[:15]
    print("top AI-tell n-grams:", ", ".join(f"{k!r}({v:+.2f})" for k, v in top))


def autogluon_rules(df):
    """Best-effort: print AutoGluon's interpretable rule models for the log.
    Never fatal — the deployed model is the sklearn pipeline above."""
    try:
        from autogluon.tabular import TabularPredictor
        pred = TabularPredictor(label="label", problem_type="binary",
                                eval_metric="f1", path="models/ag_interpretable")
        pred.fit(df, presets="interpretable", time_limit=600, verbosity=1)
        print(pred.leaderboard(silent=True).to_string())
        if hasattr(pred, "print_interpretable_models"):
            pred.print_interpretable_models()
    except Exception as e:  # autogluon optional in some envs
        print(f"[autogluon interpretable step skipped: {e}]", file=sys.stderr)


def main():
    df = pd.read_parquet("data/tropes.parquet")
    if df.label.nunique() < 2:
        print("need both classes to train; aborting", file=sys.stderr)
        sys.exit(2)
    os.makedirs("models", exist_ok=True)
    pipe = train_sklearn(df)
    joblib.dump(pipe, "models/pipeline.joblib")
    dump_explanations(pipe, "models/explain.json")
    print("wrote models/pipeline.joblib + models/explain.json")
    autogluon_rules(df)


if __name__ == "__main__":
    main()
