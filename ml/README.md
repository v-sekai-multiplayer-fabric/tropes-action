# Explainable ONNX tropes classifier

An optional, advisory machine-learning layer over the deterministic regex gate,
following the [jobs-lazy-onboarding](https://github.com/fire/jobs-lazy-onboarding)
pattern: AutoGluon trains, the model exports to ONNX, and a torch-free runtime
(`onnxruntime` + numpy) scores text on the CPU. The model is fetched lazily at
action runtime, so the base action stays a fast, dependency-free `bash` check.

## Pipeline

1. `synth_from_history.py` mines git history for labelled prose. Lines deleted
   over time that match a known tell are positives (a human removed that
   phrasing); surviving clean lines are negatives. The regex patterns mirror
   `../check_tropes.sh`, so the model starts as a generalisable soft version of
   the rules and improves as the corpus grows. Output: `data/tropes.parquet`.
2. `train_tropes.py` fits the deployed model: word + char n-gram TF-IDF into a
   `LogisticRegression`. It is inherently explainable — every n-gram has a signed
   weight (`models/explain.json`) — and fully ONNX-exportable. It also runs
   AutoGluon Tabular with `presets="interpretable"` to print rule-based models
   and a leaderboard into the run log, so the chosen signal is auditable.
3. `export_onnx.py` converts the pipeline with `skl2onnx` (the whole
   featurization is baked into the graph; the input is raw strings) and validates
   the ONNX output against sklearn before writing `onnx_tropes/model.onnx`.
4. `tropes_ml.py` is the runtime: `onnxruntime` (GPU provider if present, CPU
   otherwise) + numpy, no sklearn/torch. It scores each prose line and, because
   the model is linear, names the exact n-grams that drove a flag.

## Where it runs

`train-model.yml` does steps 1–3 in CI (monthly + on demand) and publishes
`model.onnx` + `explain.json` as the `model-latest` release. The action
downloads them only when called with `ml: true`:

```yaml
- uses: v-sekai-multiplayer-fabric/tropes-action@v1
  with:
    ml: true # advisory; never fails the build
    ml-threshold: "0.6"
```

## Run locally

```bash
pip install -r ml/requirements-train.txt
python ml/synth_from_history.py
python ml/train_tropes.py
python ml/export_onnx.py
pip install -r ml/requirements-runtime.txt
python ml/tropes_ml.py --model-dir onnx_tropes
```
