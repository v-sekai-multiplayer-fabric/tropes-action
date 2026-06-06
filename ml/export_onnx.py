# SPDX-License-Identifier: MIT
"""
Export the FT-Transformer with AutoGluon's NATIVE ONNX export
(MultiModalPredictor.export_onnx) — no skl2onnx, no hand-rolled torch.onnx.

Heavily instrumented: prints the exported graph's input/output signature and
validates onnxruntime CPU output against predictor.predict_proba, so one CI run
pins down exactly what the torch-free runtime must feed.

Outputs:
  onnx_tropes/model.onnx
  onnx_tropes/feature_names.json
  onnx_tropes/io.json           -- discovered input/output signature
  onnx_tropes/feature_importance.json (if available)
"""
import json
import os
import shutil
import numpy as np
import pandas as pd

import features as F


def main():
    from autogluon.multimodal import MultiModalPredictor
    pred = MultiModalPredictor.load("models/mm_ft")

    src = pd.read_parquet("data/tropes.parquet").head(8)
    tab = pd.DataFrame(F.matrix(src["text"].tolist()), columns=F.FEATURE_NAMES)

    os.makedirs("onnx_tropes", exist_ok=True)
    onnx_path = os.path.abspath("onnx_tropes/model.onnx")
    out = pred.export_onnx(data=tab, path=onnx_path)
    if isinstance(out, str) and os.path.exists(out):
        onnx_path = out
    print("exported:", onnx_path)

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    io = {
        "inputs": [{"name": i.name, "type": str(i.type), "shape": list(i.shape)}
                   for i in sess.get_inputs()],
        "outputs": [{"name": o.name, "type": str(o.type), "shape": list(o.shape)}
                    for o in sess.get_outputs()],
    }
    print("ONNX IO:\n" + json.dumps(io, indent=2))
    with open("onnx_tropes/io.json", "w") as fh:
        json.dump(io, fh, indent=2)

    # Best-effort validation: feed the named-feature matrix to the single float
    # input if the signature is the simple one-tensor case; otherwise just record
    # the signature for the next iteration to wire the runtime precisely.
    try:
        ref = pred.predict_proba(tab)
        ref1 = np.asarray(ref)[:, 1] if np.ndim(ref) == 2 else np.asarray(ref)
        feats = F.matrix(src["text"].tolist())
        float_inputs = [i for i in sess.get_inputs() if "float" in str(i.type)]
        if len(float_inputs) == 1 and len(sess.get_inputs()) == 1:
            res = sess.run(None, {float_inputs[0].name: feats})
            proba = next(r for r in res if getattr(r, "ndim", 0) == 2 and r.shape[1] == 2)
            max_abs = float(np.abs(proba[:, 1] - ref1).max())
            print(f"onnx-vs-predictor max_abs_diff {max_abs:.2e}")
        else:
            print("multi-input signature; runtime wiring deferred to io.json")
    except Exception as e:
        print(f"[validation note: {e}]")

    shutil.copyfile("models/feature_names.json", "onnx_tropes/feature_names.json")
    if os.path.exists("models/feature_importance.json"):
        shutil.copyfile("models/feature_importance.json", "onnx_tropes/feature_importance.json")
    print("export step done")


if __name__ == "__main__":
    main()
