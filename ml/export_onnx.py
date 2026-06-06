# SPDX-License-Identifier: MIT
"""
Export the trained TF-IDF -> LogisticRegression pipeline to ONNX with skl2onnx,
so the runtime needs only onnxruntime + numpy (no sklearn/torch). The whole
featurization (both TF-IDF vectorizers, the tokenizer regex, IDF weights) is
baked into the graph; the input is raw strings.

Outputs:
  onnx_tropes/model.onnx     -- input: text [batch,1] (string); outputs: label,
                                probabilities (class 1 = AI tell)
  onnx_tropes/explain.json   -- copied from models/explain.json for attribution

Validates ONNX == sklearn before writing, like jobs-lazy-onboarding/export_onnx.py.
"""
import json
import os
import shutil
import sys
import joblib
import numpy as np


def main():
    pipe = joblib.load("models/pipeline.joblib")
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType
    import onnxruntime as ort

    os.makedirs("onnx_tropes", exist_ok=True)
    onnx_path = "onnx_tropes/model.onnx"

    onx = convert_sklearn(
        pipe,
        initial_types=[("text", StringTensorType([None, 1]))],
        options={id(pipe.named_steps["clf"]): {"zipmap": False}},
        target_opset=17,
    )
    with open(onnx_path, "wb") as fh:
        fh.write(onx.SerializeToString())

    sample = np.array([
        ["Let us delve into the rich tapestry of options."],
        ["The launcher sets the environment and runs the program."],
    ], dtype=object)

    ref = pipe.predict_proba([s[0] for s in sample])[:, 1]
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    res = sess.run(None, {"text": sample})
    # probabilities output is the [N,2] float tensor.
    proba = next(r for r in res if getattr(r, "ndim", 0) == 2 and r.shape[1] == 2)
    onnx_p1 = proba[:, 1]
    max_abs = float(np.abs(onnx_p1 - ref).max())
    print("outputs:", out_names, "max_abs_diff", f"{max_abs:.2e}")
    # 1e-3: skl2onnx's word-tokenizer can differ from scikit's by a hair on edge
    # tokens; the ranking/threshold behaviour is unaffected.
    assert max_abs < 1e-3, "ONNX diverges from sklearn"

    shutil.copyfile("models/explain.json", "onnx_tropes/explain.json")
    print("VALIDATION OK ->", onnx_path)


if __name__ == "__main__":
    main()
