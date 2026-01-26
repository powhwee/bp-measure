"""Export the UCI StandardScaler to JSON for web deployment.

This script reads the saved model metadata containing the sklearn StandardScaler
and exports the mean and scale (std) arrays to a JSON file that can be loaded
by the web application.
"""

import json
import os
import joblib


def export_scaler():
    metadata_path = "model/bp_model_resnet_metadata.pkl"
    output_path = "web/public/uci_scaler.json"
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return
    
    print(f"Loading metadata from {metadata_path}...")
    metadata = joblib.load(metadata_path)
    
    scaler = metadata.get('scaler')
    if scaler is None:
        print("Error: No scaler found in metadata")
        return
    
    print(f"Scaler type: {type(scaler).__name__}")
    print(f"Mean shape: {scaler.mean_.shape}")
    print(f"Scale shape: {scaler.scale_.shape}")
    
    # Export to JSON
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
        "n_features": int(scaler.n_features_in_),
        "description": "UCI StandardScaler for CNN blood pressure model. Apply as: (x - mean) / std"
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(scaler_data, f, indent=2)
    
    print(f"✓ Scaler exported to {output_path}")
    print(f"  - {len(scaler_data['mean'])} mean values")
    print(f"  - {len(scaler_data['std'])} std values")


if __name__ == "__main__":
    export_scaler()
