"""
export_onnx.py — export FuelNet to ONNX for edge inference

Writes models/fuel_co2_99pct_model.onnx
Input:  (1, in_features) float32 scaled features
Output: (1,1) residual (CO2 - FC*23.7); add FC*23.7 post-hoc for CO2
"""
import pathlib
import json
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PTH = MODELS / "fuel_co2_99pct_model.pth"
ONNX = MODELS / "fuel_co2_99pct_model.onnx"

# import model def from predict (keeps single source)
import sys
sys.path.append(str(ROOT / "src"))
from predict import FuelNet

def export():
    if not PTH.exists():
        print(f"Checkpoint not found: {PTH} — train first")
        return
    ckpt = torch.load(PTH, map_location="cpu", weights_only=False)
    in_features = ckpt.get("in_features", 10)
    model = FuelNet(in_features=in_features)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(1, in_features, dtype=torch.float32)

    try:
        # classic exporter (dynamo=False) is stable; dynamo=True hangs on this MLP in torch 2.13
        torch.onnx.export(
            model,
            dummy,
            str(ONNX),
            input_names=["input"],
            output_names=["residual"],
            dynamic_axes={"input": {0: "batch"}, "residual": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        print(f"Exported ONNX to {ONNX} (in_features={in_features})")
        # verify with onnx if present
        try:
            import onnx
            m = onnx.load(str(ONNX))
            onnx.checker.check_model(m)
            print("ONNX checker: OK")
        except Exception as e:
            print(f"ONNX check skipped/failed: {e}")

        # verify runtime (onnxruntime if available)
        try:
            import onnxruntime as ort
            import numpy as np
            sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
            x = np.random.randn(1, in_features).astype(np.float32)
            y_onnx = sess.run(None, {"input": x})[0]
            with torch.no_grad():
                y_torch = model(torch.tensor(x)).numpy()
            diff = float(abs(y_onnx - y_torch).max())
            print(f"ONNX vs Torch max diff: {diff:.6f} {'OK' if diff < 1e-4 else 'WARN'}")
        except ImportError:
            print("onnxruntime not installed — install via pip install onnxruntime for verification (optional)")
        except Exception as e:
            print(f"onnxruntime verify failed: {e}")

        # also write meta sidecar
        meta = {"in_features": in_features, "opset": 17, "model": "FuelNet", "note": "Output is residual; CO2 = residual + FC*23.7"}
        if (MODELS / "model_meta.json").exists():
            try:
                base = json.loads((MODELS / "model_meta.json").read_text())
                meta.update(base)
            except:
                pass
        (MODELS / "onnx_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"Wrote {MODELS / 'onnx_meta.json'}")

    except Exception as e:
        print(f"ONNX export failed: {e}")
        raise

if __name__ == "__main__":
    export()

