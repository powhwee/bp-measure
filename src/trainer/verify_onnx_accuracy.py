import torch
import onnxruntime as ort
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.trainer.utils import load_cnn_model, WINDOW_SIZE

def verify_accuracy():
    print("=" * 60)
    print("VERIFYING ONNX EXPORT ACCURACY")
    print("=" * 60)

    # 1. Load PyTorch Model
    pt_path = "model/bp_model_resnet"
    print(f"\n[1] Loading PyTorch model from {pt_path}...")
    try:
        pt_model, _ = load_cnn_model(pt_path)
        pt_model.eval()
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        return

    # 2. Load ONNX Model
    onnx_path = "web/public/model.onnx"
    print(f"\n[2] Loading ONNX model from {onnx_path}...")
    try:
        ort_session = ort.InferenceSession(onnx_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    # 3. Generate Test Data
    print("\n[3] Generating random test inputs...")
    batch_size = 5
    # Shape: (Batch, 1, 625)
    dummy_input = torch.randn(batch_size, 1, WINDOW_SIZE, dtype=torch.float32)
    dummy_input_numpy = dummy_input.numpy()

    # 4. Run PyTorch Inference
    print("\n[4] Running PyTorch inference...")
    with torch.no_grad():
        pt_output = pt_model(dummy_input).numpy()

    # 5. Run ONNX Inference
    print("\n[5] Running ONNX inference...")
    onnx_inputs = {ort_session.get_inputs()[0].name: dummy_input_numpy}
    onnx_output = ort_session.run(None, onnx_inputs)[0]

    # 6. Compare Results
    print("\n[6] Comparing results...")
    
    # Calculate errors
    diff = np.abs(pt_output - onnx_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print("-" * 40)
    print(f"PyTorch Output (First sample):\n{pt_output[0]}")
    print(f"ONNX Output (First sample):   \n{onnx_output[0]}")
    print("-" * 40)
    print(f"Max Absolute Difference:  {max_diff:.2e}")
    print(f"Mean Absolute Difference: {mean_diff:.2e}")
    
    # Check tolerance (typically 1e-4 or 1e-5 for float32)
    tolerance = 1e-4
    if max_diff < tolerance:
        print(f"\n✅ SUCCESS: Difference is well within float32 tolerance ({tolerance}).")
        print("There is virtually NO loss of accuracy.")
    else:
        print(f"\n⚠️ WARNING: Difference exceeds typical float32 tolerance ({tolerance}).")

if __name__ == "__main__":
    verify_accuracy()
