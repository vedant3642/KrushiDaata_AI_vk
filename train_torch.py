import os
import copy
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Dataset Class
# ==========================================
class CropFertilizerDataset(Dataset):
    def __init__(self, X_num, X_dist, X_soil, y_crop, y_fert):
        self.X_num = torch.FloatTensor(X_num)
        self.X_dist = torch.LongTensor(X_dist)
        self.X_soil = torch.LongTensor(X_soil)
        self.y_crop = torch.LongTensor(y_crop)
        self.y_fert = torch.LongTensor(y_fert)
        
    def __len__(self):
        return len(self.X_num)
        
    def __getitem__(self, idx):
        return (
            self.X_num[idx],
            self.X_dist[idx],
            self.X_soil[idx],
            self.y_crop[idx],
            self.y_fert[idx]
        )

# ==========================================
# 2. Multi-Task PyTorch Model
# ==========================================
class MultiTaskModel(nn.Module):
    def __init__(self, num_districts, num_soils, num_numerical, num_crops, num_fertilizers):
        super().__init__()
        # Embedding dimensions (vocab size, embedding dim)
        self.district_embed = nn.Embedding(num_districts, 4)
        self.soil_embed = nn.Embedding(num_soils, 4)
        
        # Shared layer stack
        # 6 numerical inputs + 4 district emb + 4 soil emb = 14 total inputs
        self.shared = nn.Sequential(
            nn.Linear(num_numerical + 4 + 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Branch Heads
        self.crop_head = nn.Linear(64, num_crops)
        self.fert_head = nn.Linear(64, num_fertilizers)
        
    def forward(self, x_num, x_dist, x_soil):
        # Retrieve embeddings (shape: [batch_size, 4])
        dist_emb = self.district_embed(x_dist)
        soil_emb = self.soil_embed(x_soil)
        
        # Concatenate numerical and categorical embeddings
        x = torch.cat([x_num, dist_emb, soil_emb], dim=1)
        
        # Pass through the shared network representation
        shared_out = self.shared(x)
        
        # Get class logits for both tasks
        crop_logits = self.crop_head(shared_out)
        fert_logits = self.fert_head(shared_out)
        
        return crop_logits, fert_logits

# ==========================================
# 3. Pipeline Execution
# ==========================================
def main():
    print("Loading datasets...")
    df_real = pd.read_csv("Crop and fertilizer dataset (1).csv")
    
    # 3.1 Cleaning and Dropping columns
    print("Pre-processing and cleaning columns...")
    df_real = df_real.drop(columns=["Link"], errors="ignore")
    df_real["Soil_color"] = df_real["Soil_color"].str.strip()
    df_real["District_Name"] = df_real["District_Name"].str.strip()
    df_real["Crop"] = df_real["Crop"].str.strip()
    df_real["Fertilizer"] = df_real["Fertilizer"].str.strip()
    
    augmented_csv = "synthetic_crop_fertilizer_augmented.csv"
    if os.path.exists(augmented_csv):
        print(f"Loading augmented synthetic dataset from '{augmented_csv}'...")
        df_fake = pd.read_csv(augmented_csv)
        df_fake["Soil_color"] = df_fake["Soil_color"].str.strip()
        df_fake["District_Name"] = df_fake["District_Name"].str.strip()
        df_fake["Crop"] = df_fake["Crop"].str.strip()
        df_fake["Fertilizer"] = df_fake["Fertilizer"].str.strip()
        
        # Combine real and fake data
        df = pd.concat([df_real, df_fake], ignore_index=True)
        print(f"Data successfully combined. Total rows: {len(df)} (Real: {len(df_real)}, Synthetic: {len(df_fake)})")
    else:
        df = df_real
        print(f"Augmented dataset not found. Proceeding with real dataset only (Total rows: {len(df)})")
    
    
    # 3.2 Defining Features and Targets
    categorical_features = ["District_Name", "Soil_color"]
    numerical_features = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Rainfall", "Temperature"]
    
    # Label encode categorical input features
    print("Encoding categorical inputs...")
    district_encoder = LabelEncoder()
    df["District_Encoded"] = district_encoder.fit_transform(df["District_Name"])
    
    soil_encoder = LabelEncoder()
    df["Soil_Encoded"] = soil_encoder.fit_transform(df["Soil_color"])
    
    # Label encode target classes
    print("Encoding target labels...")
    crop_encoder = LabelEncoder()
    df["Crop_Encoded"] = crop_encoder.fit_transform(df["Crop"])
    
    fertilizer_encoder = LabelEncoder()
    df["Fertilizer_Encoded"] = fertilizer_encoder.fit_transform(df["Fertilizer"])
    
    # Create a stratification helper key
    # If crop_fertilizer combo has enough items, we use it for stratification;
    # otherwise we stratify solely on Crop.
    df["stratify_key"] = df["Crop"].astype(str) + "_" + df["Fertilizer"].astype(str)
    df["split_stratify"] = df["stratify_key"]
    
    # Map keys with < 10 members to the Crop name
    freqs = df["split_stratify"].value_counts()
    rare_keys = freqs[freqs < 10].index
    df.loc[df["split_stratify"].isin(rare_keys), "split_stratify"] = df["Crop"]
    
    # Map any remaining keys with < 10 members to 'Other_Rare'
    freqs2 = df["split_stratify"].value_counts()
    rare_keys2 = freqs2[freqs2 < 10].index
    df.loc[df["split_stratify"].isin(rare_keys2), "split_stratify"] = "Other_Rare"
    
    # Ensure 'Other_Rare' itself has >= 10 members, else fall back to the most common class
    freqs3 = df["split_stratify"].value_counts()
    if freqs3.get("Other_Rare", 0) < 10 and freqs3.get("Other_Rare", 0) > 0:
        most_common = freqs3.index[0] if freqs3.index[0] != "Other_Rare" else freqs3.index[1]
        df.loc[df["split_stratify"] == "Other_Rare", "split_stratify"] = most_common
    
    
    
    # 3.3 Train / Val / Test splits (70% / 15% / 15%)
    print("Splitting train, validation, and test sets...")
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["split_stratify"],
        random_state=42
    )
    
    # Re-stratify temp split for val and test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["split_stratify"],
        random_state=42
    )
    
    print(f"Train size: {train_df.shape[0]}, Val size: {val_df.shape[0]}, Test size: {test_df.shape[0]}")
    
    # 3.4 Scale numerical features (fit on train only!)
    print("Normalizing numerical inputs...")
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_df[numerical_features])
    X_val_num = scaler.transform(val_df[numerical_features])
    X_test_num = scaler.transform(test_df[numerical_features])
    
    # Categorical arrays
    X_train_dist = train_df["District_Encoded"].values
    X_train_soil = train_df["Soil_Encoded"].values
    
    X_val_dist = val_df["District_Encoded"].values
    X_val_soil = val_df["Soil_Encoded"].values
    
    X_test_dist = test_df["District_Encoded"].values
    X_test_soil = test_df["Soil_Encoded"].values
    
    # Target arrays
    y_train_crop = train_df["Crop_Encoded"].values
    y_train_fert = train_df["Fertilizer_Encoded"].values
    
    y_val_crop = val_df["Crop_Encoded"].values
    y_val_fert = val_df["Fertilizer_Encoded"].values
    
    y_test_crop = test_df["Crop_Encoded"].values
    y_test_fert = test_df["Fertilizer_Encoded"].values
    
    # 3.5 Calculate Class Weights for Imbalance
    print("Computing class weights...")
    unique_train_crops = np.unique(y_train_crop)
    crop_w = compute_class_weight(
        class_weight="balanced",
        classes=unique_train_crops,
        y=y_train_crop
    )
    crop_loss_weights = torch.ones(len(crop_encoder.classes_))
    for idx, c in enumerate(unique_train_crops):
        crop_loss_weights[c] = crop_w[idx]
        
    unique_train_ferts = np.unique(y_train_fert)
    fert_w = compute_class_weight(
        class_weight="balanced",
        classes=unique_train_ferts,
        y=y_train_fert
    )
    fert_loss_weights = torch.ones(len(fertilizer_encoder.classes_))
    for idx, f in enumerate(unique_train_ferts):
        fert_loss_weights[f] = fert_w[idx]
        
    # 3.6 Create PyTorch Dataloaders
    train_dataset = CropFertilizerDataset(X_train_num, X_train_dist, X_train_soil, y_train_crop, y_train_fert)
    val_dataset = CropFertilizerDataset(X_val_num, X_val_dist, X_val_soil, y_val_crop, y_val_fert)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # Initialize model
    model = MultiTaskModel(
        num_districts=len(district_encoder.classes_),
        num_soils=len(soil_encoder.classes_),
        num_numerical=len(numerical_features),
        num_crops=len(crop_encoder.classes_),
        num_fertilizers=len(fertilizer_encoder.classes_)
    )
    
    # Loss functions & Optimizer
    criterion_crop = nn.CrossEntropyLoss(weight=crop_loss_weights)
    criterion_fert = nn.CrossEntropyLoss(weight=fert_loss_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 3.7 Training Loop with Early Stopping
    epochs = 500
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 20
    patience_counter = 0
    
    print("\nStarting training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_crop_correct = 0
        train_fert_correct = 0
        total_samples = 0
        
        for batch in train_loader:
            x_num, x_dist, x_soil, y_crop, y_fert = batch
            
            optimizer.zero_grad()
            
            crop_logits, fert_logits = model(x_num, x_dist, x_soil)
            
            loss_c = criterion_crop(crop_logits, y_crop)
            loss_f = criterion_fert(fert_logits, y_fert)
            
            # Combine losses (equal weighting works well)
            loss = loss_c + loss_f
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x_num.size(0)
            
            # Track training accuracies
            _, pred_c = torch.max(crop_logits, 1)
            _, pred_f = torch.max(fert_logits, 1)
            train_crop_correct += (pred_c == y_crop).sum().item()
            train_fert_correct += (pred_f == y_fert).sum().item()
            total_samples += x_num.size(0)
            
        epoch_train_loss = train_loss / total_samples
        epoch_train_crop_acc = train_crop_correct / total_samples
        epoch_train_fert_acc = train_fert_correct / total_samples
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_crop_correct = 0
        val_fert_correct = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x_num, x_dist, x_soil, y_crop, y_fert = batch
                crop_logits, fert_logits = model(x_num, x_dist, x_soil)
                
                loss_c = criterion_crop(crop_logits, y_crop)
                loss_f = criterion_fert(fert_logits, y_fert)
                loss = loss_c + loss_f
                
                val_loss += loss.item() * x_num.size(0)
                
                _, pred_c = torch.max(crop_logits, 1)
                _, pred_f = torch.max(fert_logits, 1)
                
                val_crop_correct += (pred_c == y_crop).sum().item()
                val_fert_correct += (pred_f == y_fert).sum().item()
                val_samples += x_num.size(0)
                
        epoch_val_loss = val_loss / val_samples
        epoch_val_crop_acc = val_crop_correct / val_samples
        epoch_val_fert_acc = val_fert_correct / val_samples
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {epoch_train_loss:.4f} (Crop Acc: {epoch_train_crop_acc:.3f}, Fert Acc: {epoch_train_fert_acc:.3f}) | "
                  f"Val Loss: {epoch_val_loss:.4f} (Crop Acc: {epoch_val_crop_acc:.3f}, Fert Acc: {epoch_val_fert_acc:.3f})")
            
        # Check early stopping conditions
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Restoring best weights...")
                break
                
    # Restore best weights
    model.load_state_dict(best_model_wts)
    
    # 3.8 Model Evaluation on Test set
    print("\nEvaluating model performance on test set...")
    model.eval()
    t_num = torch.FloatTensor(X_test_num)
    t_dist = torch.LongTensor(X_test_dist)
    t_soil = torch.LongTensor(X_test_soil)
    
    with torch.no_grad():
        crop_logits, fert_logits = model(t_num, t_dist, t_soil)
        
        _, pred_crop = torch.max(crop_logits, 1)
        _, pred_fert = torch.max(fert_logits, 1)
        
    y_pred_crop = pred_crop.numpy()
    y_pred_fert = pred_fert.numpy()
    
    print("\n" + "="*50)
    print("TEST METRICS - CROP RECOMMENDATION")
    print("="*50)
    print(f"Accuracy: {accuracy_score(y_test_crop, y_pred_crop):.4f}")
    print(f"Macro F1-Score: {f1_score(y_test_crop, y_pred_crop, average='macro'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_crop, y_pred_crop, target_names=crop_encoder.classes_))
    
    print("\n" + "="*50)
    print("TEST METRICS - FERTILIZER RECOMMENDATION")
    print("="*50)
    print(f"Accuracy: {accuracy_score(y_test_fert, y_pred_fert):.4f}")
    print(f"Macro F1-Score: {f1_score(y_test_fert, y_pred_fert, average='macro'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_fert, y_pred_fert, target_names=fertilizer_encoder.classes_))
    
    # 3.9 Save Model and preprocessors to saved_models_2
    print("\nSaving final model checkpoint to saved_models_2/...")
    os.makedirs("saved_models_2", exist_ok=True)
    
    # Create complete checkpoint structure
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "mappings": {
            "districts": list(district_encoder.classes_),
            "soils": list(soil_encoder.classes_),
            "crops": list(crop_encoder.classes_),
            "fertilizers": list(fertilizer_encoder.classes_)
        },
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "features": numerical_features
        },
        "model_config": {
            "num_districts": len(district_encoder.classes_),
            "num_soils": len(soil_encoder.classes_),
            "num_numerical": len(numerical_features),
            "num_crops": len(crop_encoder.classes_),
            "num_fertilizers": len(fertilizer_encoder.classes_)
        }
    }
    
    # Determine non-conflicting output filename
    base_save_path = "saved_models_2/multitask_model"
    save_path = f"{base_save_path}.pt"
    if os.path.exists(save_path):
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = f"{base_save_path}_{timestamp}.pt"
        
    torch.save(checkpoint, save_path)
    print(f"Model and preprocessor mapping objects successfully saved in {save_path}")

if __name__ == "__main__":
    main()
