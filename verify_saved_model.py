import torch
import numpy as np
from train import MultiTaskModel

def verify():
    checkpoint_path = "saved_models_2/multitask_model.pt"
    print(f"Loading checkpoint from '{checkpoint_path}'...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print("Checkpoint successfully loaded!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Please ensure you have trained the model first by running: venv\\Scripts\\python train.py")
        return

    # 1. Load config and initialize model structure
    config = checkpoint["model_config"]
    model = MultiTaskModel(
        num_districts=config["num_districts"],
        num_soils=config["num_soils"],
        num_numerical=config["num_numerical"],
        num_crops=config["num_crops"],
        num_fertilizers=config["num_fertilizers"]
    )
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("PyTorch model structure and weights successfully loaded!")

    # 2. Define test input
    test_input = {
        "District_Name": "Kolhapur",
        "Soil_color": "Black",
        "Nitrogen": 90,
        "Phosphorus": 50,
        "Potassium": 100,
        "pH": 6.5,
        "Rainfall": 1200,
        "Temperature": 25
    }
    print(f"\nTest input features: {test_input}")

    # 3. Preprocess Categorical Inputs
    districts = checkpoint["mappings"]["districts"]
    soils = checkpoint["mappings"]["soils"]
    
    try:
        dist_idx = districts.index(test_input["District_Name"])
    except ValueError:
        raise ValueError(f"District {test_input['District_Name']} not found in trained labels: {districts}")
        
    try:
        soil_idx = soils.index(test_input["Soil_color"])
    except ValueError:
        raise ValueError(f"Soil color {test_input['Soil_color']} not found in trained labels: {soils}")

    # 4. Preprocess Numerical Inputs (using saved StandardScaler mean and scale)
    mean = np.array(checkpoint["scaler"]["mean"])
    scale = np.array(checkpoint["scaler"]["scale"])
    
    features = [
        test_input["Nitrogen"],
        test_input["Phosphorus"],
        test_input["Potassium"],
        test_input["pH"],
        test_input["Rainfall"],
        test_input["Temperature"]
    ]
    
    scaled_features = (np.array(features) - mean) / scale

    # 5. Convert to PyTorch Tensors
    x_num = torch.FloatTensor([scaled_features])
    x_dist = torch.LongTensor([dist_idx])
    x_soil = torch.LongTensor([soil_idx])

    # 6. Prediction Inference
    print("\nRunning inference...")
    with torch.no_grad():
        crop_logits, fert_logits = model(x_num, x_dist, x_soil)
        
        # Softmax probabilities
        crop_probs = torch.softmax(crop_logits, dim=1)
        fert_probs = torch.softmax(fert_logits, dim=1)
        
        # Get argmax index and confidence score
        val_c, pred_c = torch.max(crop_probs, 1)
        val_f, pred_f = torch.max(fert_probs, 1)

    crops = checkpoint["mappings"]["crops"]
    fertilizers = checkpoint["mappings"]["fertilizers"]

    predicted_crop = crops[pred_c.item()]
    predicted_fert = fertilizers[pred_f.item()]
    
    crop_conf = val_c.item()
    fert_conf = val_f.item()

    # Output verification results
    print("\n" + "="*50)
    print("VERIFICATION RESULTS")
    print("="*50)
    print(f"🌱 Recommended Crop:       {predicted_crop} (Confidence: {crop_conf*100:.2f}%)")
    print(f"🧪 Recommended Fertilizer: {predicted_fert} (Confidence: {fert_conf*100:.2f}%)")
    print("="*50)
    print("Verification successfully completed! The model loads and predicts correctly.")

if __name__ == "__main__":
    verify()
