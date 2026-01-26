import torch
import os
from src.trainer.utils import load_cnn_model, WINDOW_SIZE, create_model

def export_model():
    # 1. Load trained model
    model_path = "model/bp_model_resnet"
    if os.path.exists(model_path + ".pt"):
        print(f"Loading trained weights from {model_path}...")
        try:
            model, metadata = load_cnn_model(model_path)
            print(f"Loaded model type: {metadata.get('config', {}).get('model_type', 'Unknown')}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to random weights (ResNet1D)...")
            model = create_model('ResNet1D')
    else:
        print("Warning: Trained model not found. Using random weights (ResNet1D).")
        model = create_model('ResNet1D')

    model.eval()

    # 2. Create dummy input (Batch Size: 1, Channels: 1, Length: 625)
    dummy_input = torch.randn(1, 1, WINDOW_SIZE, requires_grad=True)

    # 3. Define output path
    # Save directly to web app's public folder
    output_dir = "web/public"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model.onnx")

    # 4. Export
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # 5. Ensure single file (merge external data if any)
    import onnx
    print("Verifying model format...")
    onnx_model = onnx.load(output_path)
    # Save again to force single file (if < 2GB)
    onnx.save(onnx_model, output_path)
    print("✓ Model saved as single file.")
    print("✓ Model exported successfully!")
    print(f"Input shape: {dummy_input.shape}")
    print("Dynamic axes: batch_size for input and output")

if __name__ == "__main__":
    export_model()
