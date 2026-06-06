# FT-Transformer ONNX tropes classifier

An optional, advisory ML layer over the deterministic regex gate, running as
**CPU ONNX inference**. It uses **interpretable deep learning**
(FT-Transformer) trained and exported entirely with **AutoGluon** — AutoGluon's
own `MultiModalPredictor.export_onnx`, so there is **no skl2onnx** and no
hand-rolled `torch.onnx`. The model is fetched lazily at action runtime.

## Interpretable + byte-general features

FT-Transformer tokenizes one token per feature, so instead of thousands of hash
buckets we use a compact, **named** feature set (`features.py`, ~35 columns):
counts of the codepoints LLMs reach for (smart quotes `’ “ ”`, em/en dashes
`— –`, ellipsis `…`, NBSP, zero-width, emoji), raw-**byte** statistics
(non-ASCII byte ratio, byte entropy, multibyte ratio — so it generalises to any
UTF-8/binary input), punctuation/surface stats, and per-rule tell counts. Every
column has a human name, so the model's attention / feature-importance is
directly interpretable ("the em-dash-count and cliché-count features drove this").

## Why FT-Transformer + AutoGluon export

`skl2onnx` is **not part of AutoGluon** and its text/byte-vectorizer path could
not convert. FT-Transformer is interpretable deep learning that AutoGluon
exports to ONNX **natively** (`MultiModalPredictor.export_onnx`), so the whole
train→export path stays inside AutoGluon and the runtime is plain
`onnxruntime` + numpy.

## Pipeline

1. `synth_from_history.py` mines git history for labelled prose (deleted
   tell-matching lines = positives; surviving clean lines = negatives) and
   augments positives with LLM-typography variants of clean lines so the model
   learns the smart-quote / dash / ellipsis / emoji tells. → `data/tropes.parquet`.
2. `features.py` — the shared named feature extractor, used identically in
   training and inference.
3. `train_tropes.py` trains the FT-Transformer with AutoGluon MultiModal and
   saves permutation feature-importance for interpretability. → `models/mm_ft/`.
4. `export_onnx.py` exports via `MultiModalPredictor.export_onnx`, prints the
   ONNX input/output signature (`io.json`), and validates against
   `predict_proba`. → `onnx_tropes/`.
5. `tropes_ml.py` is the runtime (`onnxruntime` CPU + numpy): scores each prose
   line and explains a flag via the most-important active features + typographic
   tells. Advisory — it never fails the build.

## Where it runs

`train-model.yml` does steps 1–4 in CI and publishes `model.onnx` + `explain.json`
as the `model-latest` release. The action downloads them only with `ml: true`.

## Run locally

```bash
pip install -r ml/requirements-train.txt
python ml/synth_from_history.py
python ml/train_tropes.py
python ml/export_onnx.py
pip install -r ml/requirements-runtime.txt   # onnxruntime + numpy
python ml/tropes_ml.py --model-dir onnx_tropes
```
