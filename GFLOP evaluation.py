"""
GMACs Profiler — All Thesis Models in One Run
==============================================
Place this script in the same directory as your model files:
  CNN_BiLSTM_model.py, SlowFast_model.py,
  DualHead_R2plus1D_model.py, DualBranch_model.py,
  Temporal_Attention_model.py

Then run:  python profile_all_models.py

Requirements:  pip install fvcore

Output: a table of GMACs/clip for every model.
All values are GMACs (multiply-accumulate operations, 1 MAC = 2 FLOPs).
"""

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

from fvcore.nn import FlopCountAnalysis

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — adjust only if your training configs differ
# ═══════════════════════════════════════════════════════════════════
H, W = 224, 224              # Spatial resolution (all models)
T_R2PLUS1D    = 32           # R(2+1)D-18 and all its ablation variants
T_R3D         = 16           # R3D-18 / I3D
T_BILSTM      = 60           # CNN + Bi-LSTM
T_SLOWFAST    = 32           # SlowFast fast pathway (slow = T // ALPHA)
SLOWFAST_ALPHA = 4           # SlowFast temporal stride
T_MEAN_POOLED = 60           # Mean-Pooled baseline (all clip frames)
YOLO_GFLOPS   = 8.7          # YOLOv8n at 640×640, from Ultralytics docs
YOLO_FRAMES   = 32           # Frames YOLO processes per clip at inference
# ═══════════════════════════════════════════════════════════════════


def profile(model, dummy_input, name):
    """Run fvcore FlopCountAnalysis and return GMACs, or None on failure."""
    model.eval()
    try:
        with torch.no_grad():
            fca = FlopCountAnalysis(model, dummy_input)
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            gmacs = fca.total() / 1e9
        print(f"  ✓ {name}: {gmacs:.2f} GMACs")
        return gmacs
    except Exception as e:
        print(f"  ✗ {name} FAILED: {e}")
        return None


def lstm_gmacs_analytical(input_size, hidden_size, num_layers, bidirectional, seq_len):
    """
    Compute LSTM gate MACs analytically.
    fvcore often undercounts recurrent ops, so we add this separately.
    Formula per timestep per direction: 4 * hidden * (input + hidden)
    """
    dirs = 2 if bidirectional else 1
    total_macs = 0
    current_input = input_size
    for _ in range(num_layers):
        per_dir = 4 * hidden_size * (current_input + hidden_size)
        total_macs += per_dir * dirs * seq_len
        current_input = hidden_size * dirs   # next layer input = bidir output
    return total_macs / 1e9


def main():
    results = {}
    params  = {}

    def count_params(model, name):
        p = sum(p.numel() for p in model.parameters()) / 1e6
        params[name] = p

    # ─────────────────────────────────────────────────────────
    # 1. R(2+1)D-18 BASELINE (Challenger 3)
    # ─────────────────────────────────────────────────────────
    print("\n[1/10] R(2+1)D-18 baseline...")
    from torchvision.models.video import r2plus1d_18
    model = r2plus1d_18(num_classes=5)
    dummy = torch.randn(1, 3, T_R2PLUS1D, H, W)
    results["R(2+1)D-18"] = profile(model, dummy, "R(2+1)D-18")
    count_params(model, "R(2+1)D-18")
    del model

    # ─────────────────────────────────────────────────────────
    # 2. R3D-18 / I3D (Challenger 2)
    # ─────────────────────────────────────────────────────────
    print("\n[2/10] R3D-18 (I3D)...")
    from torchvision.models.video import r3d_18
    model = r3d_18(num_classes=5)
    dummy = torch.randn(1, 3, T_R3D, H, W)
    results["R3D-18"] = profile(model, dummy, "R3D-18")
    count_params(model, "R3D-18")
    del model

    # ─────────────────────────────────────────────────────────
    # 3. CNN + Bi-LSTM (Challenger 1)
    # ─────────────────────────────────────────────────────────
    print("\n[3/10] CNN + Bi-LSTM...")
    from CNN_BiLSTM_model import VolleyballCNNBiLSTMModel
    model = VolleyballCNNBiLSTMModel(num_classes=5)
    dummy = torch.randn(1, T_BILSTM, 3, H, W)
    cnn_cost = profile(model, dummy, "CNN+Bi-LSTM (fvcore)")
    # Add LSTM gates analytically (fvcore typically undercounts them)
    lstm_cost = lstm_gmacs_analytical(
        input_size=512, hidden_size=256,
        num_layers=2, bidirectional=True, seq_len=T_BILSTM
    )
    if cnn_cost is not None:
        results["CNN+Bi-LSTM"] = cnn_cost + lstm_cost
        print(f"    + LSTM gates (analytical): {lstm_cost:.3f} GMACs")
        print(f"    = Total: {results['CNN+Bi-LSTM']:.2f} GMACs")
    count_params(model, "CNN+Bi-LSTM")
    del model

    # ─────────────────────────────────────────────────────────
    # 4. SlowFast R50 (Literature Standard)
    # Uses torch.hub — weights should be cached from training.
    # ─────────────────────────────────────────────────────────
    print("\n[4/10] SlowFast R50...")
    try:
        from SlowFast_model import VolleyballSlowFastModel
        model = VolleyballSlowFastModel(
            num_classes=5, freeze_backbone=False, alpha=SLOWFAST_ALPHA
        )
        dummy = torch.randn(1, T_SLOWFAST, 3, H, W)
        results["SlowFast"] = profile(model, dummy, "SlowFast")
        count_params(model, "SlowFast")
        del model
    except Exception as e:
        print(f"  ✗ SlowFast import/load failed: {e}")
        print("    If torch.hub times out, set the value manually from the")
        print("    PyTorchVideo docs (~65 GMACs for SlowFast-R50 8×8 at 224²)")
        results["SlowFast"] = None

    # ─────────────────────────────────────────────────────────
    # 5. DualHead (Ablation 1)
    # ─────────────────────────────────────────────────────────
    print("\n[5/10] DualHead...")
    from DualHead_R2plus1D_model import VolleyballR2Plus1DDualHeadModel
    model = VolleyballR2Plus1DDualHeadModel(freeze_backbone=False)
    dummy = torch.randn(1, T_R2PLUS1D, 3, H, W)
    results["DualHead"] = profile(model, dummy, "DualHead")
    count_params(model, "DualHead")
    del model

    # ─────────────────────────────────────────────────────────
    # 6. ROI Cropping (Ablation 2)
    # Classifier = identical R(2+1)D-18 (same input size).
    # Full pipeline adds YOLOv8n detection on each frame.
    # ─────────────────────────────────────────────────────────
    print("\n[6/10] ROI Cropping...")
    yolo_gmacs_per_frame = YOLO_GFLOPS / 2   # Ultralytics reports FLOPs; /2 = GMACs
    yolo_total = yolo_gmacs_per_frame * YOLO_FRAMES
    r2p1d_cost = results.get("R(2+1)D-18", 325.16)
    results["ROI Cropping"] = r2p1d_cost + yolo_total
    params["ROI Cropping"] = params.get("R(2+1)D-18", 31.5) + 3.2  # YOLOv8n ≈ 3.2M
    print(f"  ✓ ROI Cropping: {results['ROI Cropping']:.2f} GMACs")
    print(f"    = R(2+1)D-18 ({r2p1d_cost:.2f}) + YOLOv8n ({yolo_total:.2f})")

    # ─────────────────────────────────────────────────────────
    # 7. DualBranch (Ablation 3)
    # ─────────────────────────────────────────────────────────
    print("\n[7/10] DualBranch...")
    from DualBranch_model import FineGrainedDualStreamModel
    model = FineGrainedDualStreamModel(num_classes=5, freeze_backbone=False)
    dummy = torch.randn(1, T_R2PLUS1D, 3, H, W)
    results["DualBranch"] = profile(model, dummy, "DualBranch")
    count_params(model, "DualBranch")
    del model

    # ─────────────────────────────────────────────────────────
    # 8. Temporal Attention (Ablation 4)
    # ─────────────────────────────────────────────────────────
    print("\n[8/10] Temporal Attention...")
    from Temporal_Attention_model import R2Plus1DTemporalAttention
    model = R2Plus1DTemporalAttention(num_classes=5, freeze_backbone=False)
    dummy = torch.randn(1, T_R2PLUS1D, 3, H, W)
    results["Temporal Attention"] = profile(model, dummy, "Temporal Attention")
    count_params(model, "Temporal Attention")
    del model

    # ─────────────────────────────────────────────────────────
    # 9. Frozen-Feature Baselines (analytical from ResNet-18)
    # ─────────────────────────────────────────────────────────
    print("\n[9/10] Frozen-feature baselines...")
    import torchvision.models as image_models
    resnet = image_models.resnet18()
    dummy_frame = torch.randn(1, 3, H, W)
    per_frame = profile(resnet, dummy_frame, "ResNet-18 (single frame)")
    if per_frame is not None:
        results["Central Frame"]  = round(per_frame * 1, 2)
        results["Mean-Pooled"]    = round(per_frame * T_MEAN_POOLED, 2)
        params["Central Frame"]   = 11.69
        params["Mean-Pooled"]     = 11.69
        print(f"    Central Frame  (1 frame):  {results['Central Frame']:.2f} GMACs")
        print(f"    Mean-Pooled    ({T_MEAN_POOLED} frames): {results['Mean-Pooled']:.2f} GMACs")
    del resnet

    # ─────────────────────────────────────────────────────────
    # 10. Heuristic baselines (no computation)
    # ─────────────────────────────────────────────────────────
    print("\n[10/10] Heuristic baselines...")
    results["Zero-R"]      = 0.0
    results["Stratified"]  = 0.0
    params["Zero-R"]       = 0.0
    params["Stratified"]   = 0.0
    print("  ✓ Zero-R:      0.00 GMACs (no forward pass)")
    print("  ✓ Stratified:  0.00 GMACs (no forward pass)")

    # ═════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═════════════════════════════════════════════════════════
    display_order = [
        ("Zero-R",              "Baseline"),
        ("Stratified",          "Baseline"),
        ("Central Frame",       "Baseline"),
        ("Mean-Pooled",         "Baseline"),
        ("SlowFast",            "Baseline"),
        ("CNN+Bi-LSTM",         "Challenger"),
        ("R3D-18",              "Challenger"),
        ("R(2+1)D-18",          "Challenger"),
        ("DualHead",            "Ablation"),
        ("ROI Cropping",        "Ablation"),
        ("DualBranch",          "Ablation"),
        ("Temporal Attention",  "Ablation"),
    ]

    print("\n" + "=" * 65)
    print(f"  {'Model':<23} {'GMACs/clip':>10} {'Params (M)':>11}  {'Category'}")
    print("  " + "-" * 61)

    for name, category in display_order:
        val = results.get(name)
        par = params.get(name)
        val_str = f"{val:>10.2f}" if val is not None else f"{'FAILED':>10}"
        par_str = f"{par:>10.1f}" if par is not None else f"{'?':>10}"
        print(f"  {name:<23} {val_str} {par_str}   {category}")

    print("  " + "-" * 61)
    print("=" * 65)

    print("\nNotes:")
    print("  • All values are GMACs (1 MAC ≈ 2 FLOPs).")
    print("  • CNN+Bi-LSTM includes analytically computed LSTM gate cost.")
    print(f"  • ROI Cropping includes YOLOv8n detector"
          f" ({YOLO_FRAMES} frames × {yolo_gmacs_per_frame:.2f} GMACs/frame).")
    print("  • Zero-R and Stratified perform no forward pass (0 GMACs).")
    print("  • SlowFast loaded via torch.hub (weights must be cached).")

    # Save to a text file for easy copy-paste
    with open("gmacs_results.txt", "w") as f:
        f.write("Model,GMACs,Params_M,Category\n")
        for name, category in display_order:
            val = results.get(name, -1)
            par = params.get(name, -1)
            f.write(f"{name},{val:.2f},{par:.1f},{category}\n")
    print("\nResults also saved to: gmacs_results.txt")


if __name__ == "__main__":
    main()