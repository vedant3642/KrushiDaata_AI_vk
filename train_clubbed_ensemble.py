import os
import copy
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Agronomic Dataset & PyTorch Model Definition
# ==========================================
class AgronomicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class AgronomicCropModel(nn.Module):
    def __init__(self, num_features=7, num_classes=22):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. Canonical Crop Mapping Dictionary
# Maps between regional names (e.g. Gram, Tur) and agronomic names (chickpea, pigeonpeas)
# ==========================================
REGIONAL_TO_AGRONOMIC_MAP = {
    "cotton": "cotton",
    "ginger": None,
    "gram": "chickpea",
    "grapes": "grapes",
    "groundnut": None,
    "jowar": None,
    "maize": "maize",
    "masoor": "lentil",
    "moong": "mungbean",
    "rice": "rice",
    "soybean": None,
    "sugarcane": None,
    "tur": "pigeonpeas",
    "turmeric": None,
    "urad": "blackgram",
    "wheat": None
}

AGRONOMIC_TO_REGIONAL_MAP = {v: k for k, v in REGIONAL_TO_AGRONOMIC_MAP.items() if v is not None}

# ==========================================
# 3. Training the Agronomic Crop Model
# ==========================================
def train_agronomic_model(csv_path="Crop_recommendation.csv", save_dir="saved_models_clubbed"):
    print("=" * 60)
    print("STEP 1: Training Agronomic Crop Model on Crop_recommendation.csv")
    print("=" * 60)
    
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    feature_cols = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]
    X = df[feature_cols].values
    y_raw = df["label"].str.strip().str.lower().values
    
    crop_encoder = LabelEncoder()
    y = crop_encoder.fit_transform(y_raw)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    train_dataset = AgronomicDataset(X_train_scaled, y_train)
    val_dataset = AgronomicDataset(X_val_scaled, y_val)
    test_dataset = AgronomicDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = AgronomicCropModel(num_features=len(feature_cols), num_classes=len(crop_encoder.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    best_weights = None
    epochs = 60
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_b)
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                logits = model(x_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * len(x_b)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_b).sum().item()
        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_dataset)
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
    model.load_state_dict(best_weights)
    model.eval()
    
    # Test evaluation
    with torch.no_grad():
        test_logits = model(torch.FloatTensor(X_test_scaled))
        test_preds = torch.argmax(test_logits, dim=1).numpy()
    
    acc = accuracy_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds, average='macro')
    print(f"\nAgronomic Model Test Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    
    os.makedirs(save_dir, exist_ok=True)
    agronomic_checkpoint = {
        "model_state_dict": model.state_dict(),
        "classes": list(crop_encoder.classes_),
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "features": feature_cols
        },
        "model_config": {
            "num_features": len(feature_cols),
            "num_classes": len(crop_encoder.classes_)
        }
    }
    
    save_path = os.path.join(save_dir, "agronomic_crop_model.pt")
    torch.save(agronomic_checkpoint, save_path)
    print(f"Saved Agronomic model to: {save_path}")
    return agronomic_checkpoint

# ==========================================
# 4. Clubbing with Original Regional Model into saved_models_clubbed/
# ==========================================
def club_and_save_ensemble(
    regional_model_path="saved_models_2/multitask_model.pt",
    save_dir="saved_models_clubbed",
    district_weight=0.72  # Prioritizing district & soil data reliability (72% weight)
):
    print("\n" + "=" * 60)
    print("STEP 2: Packaging Clubbed Ensemble in dedicated folder")
    print("=" * 60)
    
    if not os.path.exists(regional_model_path):
        raise FileNotFoundError(f"Regional model not found at {regional_model_path}")
        
    regional_checkpoint = torch.load(regional_model_path, map_location="cpu", weights_only=False)
    
    # Save a clean copy of the regional model in the new folder
    regional_save_path = os.path.join(save_dir, "regional_multitask_model.pt")
    torch.save(regional_checkpoint, regional_save_path)
    print(f"Saved Regional Multi-Task model copy to: {regional_save_path}")
    
    # Create master ensemble metadata
    ensemble_metadata = {
        "fusion_strategy": "Weighted_Confidence_Stacking",
        "description": "Dual-Stream Expert Ensemble combining Regional District Multi-Task Model and Pure Agronomic Crop Model",
        "weights": {
            "district_regional_weight": district_weight,
            "agronomic_crop_weight": round(1.0 - district_weight, 2)
        },
        "regional_model": {
            "checkpoint_file": "regional_multitask_model.pt",
            "districts": regional_checkpoint["mappings"]["districts"],
            "soils": regional_checkpoint["mappings"]["soils"],
            "crops": regional_checkpoint["mappings"]["crops"],
            "fertilizers": regional_checkpoint["mappings"]["fertilizers"],
            "scaler": regional_checkpoint["scaler"]
        },
        "agronomic_model": {
            "checkpoint_file": "agronomic_crop_model.pt",
            "crops": json.load(open(os.path.join(save_dir, "agronomic_metadata.json")))["classes"] if os.path.exists(os.path.join(save_dir, "agronomic_metadata.json")) else []
        },
        "mappings": {
            "regional_to_agronomic": REGIONAL_TO_AGRONOMIC_MAP,
            "agronomic_to_regional": AGRONOMIC_TO_REGIONAL_MAP
        }
    }
    
    meta_path = os.path.join(save_dir, "ensemble_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(ensemble_metadata, f, indent=4)
    print(f"Saved Ensemble Metadata configuration to: {meta_path}")
    print("\n[SUCCESS] Previous models in saved_models_2/ remain 100% untouched.")

# ==========================================
# 5. Main Execution
# ==========================================
def main():
    save_dir = "saved_models_clubbed"
    
    # 1. Train Agronomic Model
    agronomic_ckpt = train_agronomic_model(
        csv_path="Crop_recommendation.csv",
        save_dir=save_dir
    )
    
    # Save standalone agronomic metadata for easy inspection
    with open(os.path.join(save_dir, "agronomic_metadata.json"), "w") as f:
        json.dump({
            "classes": agronomic_ckpt["classes"],
            "scaler": agronomic_ckpt["scaler"],
            "model_config": agronomic_ckpt["model_config"]
        }, f, indent=4)
        
    # 2. Club with Regional District Model
    club_and_save_ensemble(
        regional_model_path="saved_models_2/multitask_model.pt",
        save_dir=save_dir,
        district_weight=0.72  # Prioritize highly reliable district dataset
    )
    print("\nAll clubbed models and configurations are successfully stored in 'saved_models_clubbed/'.")

if __name__ == "__main__":
    main()
