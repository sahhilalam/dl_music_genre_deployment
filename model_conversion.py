import torch
import joblib

# Load trained PyTorch weights
state_dict = torch.load(
    "ResNet50_1.pth",
    map_location="cpu"
)

# Save only the trained weights
joblib.dump(state_dict, "model.pkl")

print("model.pkl created successfully!")