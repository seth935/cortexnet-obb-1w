"""CortexNet-OBB-1W (v2) — Local Pre-Submission & Diagnostic Self-Audit Script."""

import json
import os
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort


def verify_all(repo_dir: str | Path = "."):
    root = Path(repo_dir).resolve()
    with open(root / "model.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["source_type"] == "original", "Expected source_type='original'"
    rt = manifest["runtime"]
    engine = rt["engine"] if isinstance(rt, dict) else rt
    threads = rt.get("threads", 4) if isinstance(rt, dict) else manifest.get("threads", 4)
    assert engine == "onnxruntime", "Engine must be onnxruntime"

    model_rel = manifest["detector"]["model_path"]
    model_path = root / model_rel
    assert model_path.exists(), f"Missing {model_path}"

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    assert 0.06 <= size_mb <= 5.0, f"Artifact size {size_mb:.3f} MB outside expected 60KB..5MB range"

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    h, w = manifest["detector"]["input_size"]

    # 1. Verify Empty / Flat-Field Control Frames Naturally Emit 0 Detections Above Threshold
    conf_thr = manifest["detector"]["conf_threshold"]
    for label, val in [("all-black", 0.0), ("all-white", 1.0), ("flat-gray", 0.5)]:
        canary = np.full((1, 3, h, w), val, dtype=np.float32)
        out = sess.run(None, {in_name: canary})[0]
        assert out.shape == (1, 100, 8), f"Unexpected shape: {out.shape}"
        active_col5 = int(np.sum(out[0, :, 5] >= conf_thr))
        active_col6 = int(np.sum((out[0, :, 5] >= conf_thr) & (out[0, :, 6] >= conf_thr)))
        assert active_col5 == 0 and active_col6 == 0, f"Failed empty-frame control on {label}: {active_col5} false alarms"

    # 2. Verify Active Textured Scene & 30-Frame Throughput
    scene = np.zeros((1, 3, h, w), dtype=np.float32)
    scene[:, :, 100:180, 120:220] = 0.85  # High-contrast target block
    t0 = time.perf_counter()
    for _ in range(30):
        out = sess.run(None, {in_name: scene})[0]
    avg_ms = ((time.perf_counter() - t0) / 30.0) * 1000.0
    fps = 1000.0 / avg_ms
    active_dets = int(np.sum(out[0, :, 5] >= conf_thr))
    assert active_dets >= 2, f"Expected >=2 active detections on textured scene, got {active_dets}"

    print(f"[PASS] Artifact Size: {size_mb * 1024:.1f} KB ({size_mb:.3f} MB) | Output Shape: {out.shape}")
    print(f"[PASS] Empty-Scene False Alarms: 0.0000 FPPI | Active Detections: {active_dets}")
    print(f"[PASS] Host Latency: {avg_ms:.2f} ms ({fps:.1f} FPS)")


if __name__ == "__main__":
    verify_all(Path(__file__).resolve().parent)
