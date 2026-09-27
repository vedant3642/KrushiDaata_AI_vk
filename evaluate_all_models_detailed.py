import os
import time
import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tensorflow as tf
from services.clubbed_prediction_service import ClubbedEnsemblePredictor
from train_torch import MultiTaskModel

def safe_index(lst, val):
    try:
        return lst.index(val)
    except ValueError:
        return 0

def evaluate_torch_model(checkpoint_path, df_test):
    if not os.path.exists(checkpoint_path):
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = MultiTaskModel(
        num_districts=config["num_districts"],
        num_soils=config["num_soils"],
        num_numerical=config["num_numerical"],
        num_crops=config["num_crops"],
        num_fertilizers=config["num_fertilizers"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    districts = checkpoint["mappings"]["districts"]
    soils = checkpoint["mappings"]["soils"]
    crops = checkpoint["mappings"]["crops"]
    fertilizers = checkpoint["mappings"]["fertilizers"]
    
    x_dist = torch.LongTensor([safe_index(districts, d) for d in df_test["District_Name"]])
    x_soil = torch.LongTensor([safe_index(soils, s) for s in df_test["Soil_color"]])
    
    mean = np.array(checkpoint["scaler"]["mean"])
    scale = np.array(checkpoint["scaler"]["scale"])
    numerical_cols = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Rainfall", "Temperature"]
    X_num_raw = df_test[numerical_cols].values
    X_num_scaled = (X_num_raw - mean) / scale
    x_num = torch.FloatTensor(X_num_scaled)
    
    start_t = time.time()
    with torch.no_grad():
        crop_logits, fert_logits = model(x_num, x_dist, x_soil)
        _, pred_c = torch.max(crop_logits, 1)
        _, pred_f = torch.max(fert_logits, 1)
    latency = (time.time() - start_t) / len(df_test) * 1000.0  # ms per sample
    
    y_pred_crop = pred_c.numpy()
    y_pred_fert = pred_f.numpy()
    
    y_true_crop = np.array([safe_index(crops, c) for c in df_test["Crop"]])
    y_true_fert = np.array([safe_index(fertilizers, f) for f in df_test["Fertilizer"]])
    
    return {
        "crop_acc": accuracy_score(y_true_crop, y_pred_crop),
        "crop_f1": f1_score(y_true_crop, y_pred_crop, average="macro"),
        "fert_acc": accuracy_score(y_true_fert, y_pred_fert),
        "fert_f1": f1_score(y_true_fert, y_pred_fert, average="macro"),
        "latency_ms": latency
    }

def evaluate_keras_model(model_path, metadata_path, df_test):
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    model = tf.keras.models.load_model(model_path)
    
    districts = metadata["mappings"]["districts"]
    soils = metadata["mappings"]["soils"]
    crops = metadata["mappings"]["crops"]
    fertilizers = metadata["mappings"]["fertilizers"]
    
    X_dist = np.array([safe_index(districts, d) for d in df_test["District_Name"]])
    X_soil = np.array([safe_index(soils, s) for s in df_test["Soil_color"]])
    
    mean = np.array(metadata["scaler"]["mean"])
    scale = np.array(metadata["scaler"]["scale"])
    numerical_cols = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Rainfall", "Temperature"]
    X_num_raw = df_test[numerical_cols].values
    X_num_scaled = (X_num_raw - mean) / scale
    
    start_t = time.time()
    predictions = model.predict({
        "numerical": X_num_scaled,
        "district": X_dist,
        "soil": X_soil
    }, verbose=0)
    latency = (time.time() - start_t) / len(df_test) * 1000.0  # ms per sample
    
    crop_probs, fert_probs = predictions
    y_pred_crop = np.argmax(crop_probs, axis=1)
    y_pred_fert = np.argmax(fert_probs, axis=1)
    
    y_true_crop = np.array([safe_index(crops, c) for c in df_test["Crop"]])
    y_true_fert = np.array([safe_index(fertilizers, f) for f in df_test["Fertilizer"]])
    
    return {
        "crop_acc": accuracy_score(y_true_crop, y_pred_crop),
        "crop_f1": f1_score(y_true_crop, y_pred_crop, average="macro"),
        "fert_acc": accuracy_score(y_true_fert, y_pred_fert),
        "fert_f1": f1_score(y_true_fert, y_pred_fert, average="macro"),
        "latency_ms": latency
    }

def evaluate_ensemble(model_dir, df_test, district_weight=0.72):
    predictor = ClubbedEnsemblePredictor(model_dir=model_dir, district_weight=district_weight)
    y_pred_crop = []
    y_pred_fert = []
    y_true_crop = []
    y_true_fert = []
    
    start_t = time.time()
    for _, row in df_test.iterrows():
        res = predictor.predict(
            district=row["District_Name"],
            soil=row["Soil_color"],
            nitrogen=row["Nitrogen"],
            phosphorus=row["Phosphorus"],
            potassium=row["Potassium"],
            ph=row["pH"],
            rainfall=row["Rainfall"],
            temperature=row["Temperature"],
            humidity=70.0
        )
        y_pred_crop.append(res["recommended_crop"])
        y_pred_fert.append(res["recommended_fertilizer"])
        y_true_crop.append(row["Crop"])
        y_true_fert.append(row["Fertilizer"])
    latency = (time.time() - start_t) / len(df_test) * 1000.0
    
    return {
        "crop_acc": accuracy_score(y_true_crop, y_pred_crop),
        "crop_f1": f1_score(y_true_crop, y_pred_crop, average="macro"),
        "fert_acc": accuracy_score(y_true_fert, y_pred_fert),
        "fert_f1": f1_score(y_true_fert, y_pred_fert, average="macro"),
        "latency_ms": latency
    }

def evaluate_agronomic_standalone(checkpoint_path, csv_path):
    if not os.path.exists(checkpoint_path):
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    from train_clubbed_ensemble import AgronomicCropModel
    model = AgronomicCropModel(
        num_features=checkpoint["model_config"]["num_features"],
        num_classes=checkpoint["model_config"]["num_classes"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    feature_cols = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]
    X = df[feature_cols].values
    classes = checkpoint["classes"]
    y_true = np.array([safe_index(classes, c) for c in df["label"].str.strip().str.lower()])
    
    mean = np.array(checkpoint["scaler"]["mean"])
    scale = np.array(checkpoint["scaler"]["scale"])
    X_scaled = (X - mean) / scale
    
    start_t = time.time()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_scaled))
        preds = torch.argmax(logits, dim=1).numpy()
    latency = (time.time() - start_t) / len(df) * 1000.0
    
    return {
        "crop_acc": accuracy_score(y_true, preds),
        "crop_f1": f1_score(y_true, preds, average="macro"),
        "latency_ms": latency
    }

def main():
    print("Preparing hold-out test set from Regional Dataset...")
    df = pd.read_csv("Crop and fertilizer dataset (1).csv")
    df = df.drop(columns=["Link"], errors="ignore")
    df["Soil_color"] = df["Soil_color"].str.strip()
    df["District_Name"] = df["District_Name"].str.strip()
    
    df["stratify_key"] = df["Crop"].astype(str) + "_" + df["Fertilizer"].astype(str)
    freqs = df["stratify_key"].value_counts()
    rare_combos = freqs[freqs < 3].index
    df["split_stratify"] = df["stratify_key"].apply(lambda x: x if x not in rare_combos else x.split("_")[0])
    
    _, temp_df = train_test_split(df, test_size=0.30, stratify=df["split_stratify"], random_state=42)
    _, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["split_stratify"], random_state=42)
    
    results = {}
    
    # 1. Models in saved_models_2
    results["[saved_models_2] PyTorch (multitask_model.pt)"] = evaluate_torch_model(
        "saved_models_2/multitask_model.pt", test_df
    )
    results["[saved_models_2] Keras Baseline (multitask_model.keras)"] = evaluate_keras_model(
        "saved_models_2/multitask_model.keras", "saved_models_2/multitask_model_metadata.json", test_df
    )
    results["[saved_models_2] Keras Best Regional (multitask_model0.keras)"] = evaluate_keras_model(
        "saved_models_2/multitask_model0.keras", "saved_models_2/multitask_model_metadata0.json", test_df
    )
    results["[saved_models_2] Keras Real-Only (multitask_model_real.keras)"] = evaluate_keras_model(
        "saved_models_2/multitask_model_real.keras", "saved_models_2/multitask_model_metadata_real.json", test_df
    )
    
    # 2. Standalone Regional Model in saved_models_clubbed
    results["[saved_models_clubbed] Regional Standalone (multitask_model0.keras)"] = evaluate_keras_model(
        "saved_models_clubbed/multitask_model0.keras", "saved_models_clubbed/multitask_model_metadata0.json", test_df
    )
    
    # 3. New Clubbed Dual-Stream Ensemble (multitask_model0.keras + agronomic_crop_model.pt)
    results["[saved_models_clubbed] Dual-Stream Ensemble (Keras multi_model0 + Agronomic)"] = evaluate_ensemble(
        "saved_models_clubbed", test_df, district_weight=0.72
    )
    
    # 4. Agronomic Model Standalone Performance
    agro_res = evaluate_agronomic_standalone(
        "saved_models_clubbed/agronomic_crop_model.pt", "Crop_recommendation.csv"
    )
    
    print("\n" + "="*125)
    print(f"{'COMPREHENSIVE BENCHMARK EVALUATION: ALL MODELS IN ALL FOLDERS':^125}")
    print("="*125)
    header = f"{'Model Name / Configuration':<65} | {'Crop Acc':<10} | {'Crop F1':<10} | {'Fert Acc':<10} | {'Fert F1':<10} | {'Latency':<10}"
    print(header)
    print("-" * 125)
    
    for name, m in results.items():
        if m:
            print(f"{name:<65} | {m['crop_acc']*100:>8.2f}% | {m['crop_f1']:>10.4f} | {m['fert_acc']*100:>8.2f}% | {m['fert_f1']:>10.4f} | {m['latency_ms']:>6.2f} ms")
        else:
            print(f"{name:<65} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
            
    print("="*125)
    if agro_res:
        print(f"[saved_models_clubbed] Pure Agronomic Crop Model (Standalone on 2200-sample 22-Crop Dataset):")
        print(f"  • Test Accuracy: {agro_res['crop_acc']*100:.2f}% | Macro F1: {agro_res['crop_f1']:.4f} | Avg Latency: {agro_res['latency_ms']:.2f} ms")
    print("="*125)

if __name__ == "__main__":
    main()
