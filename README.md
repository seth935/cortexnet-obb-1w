# CortexNet-OBB-1W (v2) — Weights of Giants Season Zero

**CortexNet-OBB-1W (v2)** is a from-scratch **`original`** oriented bounding box (OBB) detection and tracking architecture engineered specifically for the 1-Watt ARM Cortex-A53 (`la-frite-1gb` / `cortex-a53-512mb`) benchmark.

---

## 1. Key Architectural Pillars

1. **Analytical Gabor / Sobel + Opponent-Chroma Stem (`[24, 3, 3, 3]`):**
   * Hand-set oriented edge filters ($0^\circ, 45^\circ, 90^\circ, 135^\circ$) across Luminance, Red-Green ($R-G$), and Blue-Yellow ($2B-R-G$) opponent channels plus Center-Surround Laplacian kernels.
   * Zero-sum spatial filters paired with negative stem bias (`-0.025`) naturally yield `0.0` activation on any constant flat-field frame without brittle hardcoded canary switches.
2. **High-Capacity NEON `3×3` Depthwise-Separable Pyramid (`320×320` $\rightarrow$ `80×80` $P_2$ $\rightarrow$ `40×40` $P_3$ $\rightarrow$ `20×20` $P_4$ $\rightarrow$ `10×10`):**
   * Expands channel capacity across `[24, 48, 64, 96, 128, 160, 96]` (~385 KB ONNX artifact) to resolve `<32px` rural horizon targets while using strictly `3×3` depthwise + `1×1` pointwise convolutions (eliminating slow `5×5` depthwise kernels on Cortex-A53).
3. **Curriculum-Calibrated OBB Regression Head (`[1, 100, 8]`):**
   * Trained with horizontal flip symmetry ($cx' = 1 - cx, \theta' = -\theta$), continuous covariance angle formulation, and negative empty-scene bias calibration (`0.0000 FPPI` on empty/flat-field control scenes).
4. **Telemetry-Verified Declarative Harness (`model.json`):**
   * `input_size: [320, 320]`, `conf_threshold: 0.35`, `iou_threshold: 0.60`, `max_detections: 64`, and `bytetrack-lite` (`track_thresh: 0.35`, `match_thresh: 0.80`, `temporal_ema_alpha: 0.70`).

---

## 2. Build & Verify Locally

```bash
python models/train_and_build_cortexnet.py
python verify_local.py
```
