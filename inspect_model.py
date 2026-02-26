import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from phypush_transformer import PhysicsTransformerEstimator

# ==========================================
# DEPLOYMENT INSPECTION SCRIPT
# ==========================================

PATH = 'trained_models/transformer_epoch500.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"--- Inspecting: {PATH} ---")

# Load and Check Weights
state_dict = torch.load(PATH, map_location=device)
print(f"Model Checkpoint Loaded. Number of parameter tensors: {len(state_dict)}")

# Initialize model (using d_model=32 and version=4 from your logs)
model = PhysicsTransformerEstimator(input_dim=1, d_model=64, seq_len=60, version=4)

try:
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print("✅ Model weights loaded successfully into PhysicsTransformerEstimator (V4).")
except Exception as e:
    print(f"❌ Weight mismatch: {e}")

# Create Dummy Input
# Your model expects sequence_length=20, input_dim=1 for acc and 1 for vel
dummy_acc = torch.randn(1, 60, 1).to(device)
dummy_vel = torch.randn(1, 60, 1).to(device)

print("\n--- 🚀 Running Dummy Inference ---")
with torch.no_grad():
    # prediction = model(dummy_acc, dummy_vel)
    prediction = model(dummy_vel)

mass_out = prediction[0, 0].item()
mu_out = prediction[0, 1].item()

print(f"Input Shape (Vel): {dummy_vel.shape}")
print(f"Output Raw Tensor:     {prediction}")
print(f"Parsed Prediction:")
print(f"  -> Predicted Mass: {mass_out:.4f} kg")
print(f"  -> Predicted Mu:   {mu_out:.4f}")



# -------------------