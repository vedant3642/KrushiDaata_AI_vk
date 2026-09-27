import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score
import tensorflow as tf

def main():
    print("Loading datasets...")
    df_real = pd.read_csv("Crop and fertilizer dataset (1).csv")
    
    # 1. Clean Data
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
    
    
    # 2. Features and Targets
    categorical_features = ["District_Name", "Soil_color"]
    numerical_features = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Rainfall", "Temperature"]
    
    # Encode categorical input features
    print("Encoding categorical inputs...")
    district_encoder = LabelEncoder()
    df["District_Encoded"] = district_encoder.fit_transform(df["District_Name"])
    
    soil_encoder = LabelEncoder()
    df["Soil_Encoded"] = soil_encoder.fit_transform(df["Soil_color"])
    
    # Encode targets
    print("Encoding target labels...")
    crop_encoder = LabelEncoder()
    df["Crop_Encoded"] = crop_encoder.fit_transform(df["Crop"])
    
    fertilizer_encoder = LabelEncoder()
    df["Fertilizer_Encoded"] = fertilizer_encoder.fit_transform(df["Fertilizer"])
    
    # Create stratification key
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
    
    
    
    # 3. Splits (70 / 15 / 15)
    print("Splitting train, validation, and test sets...")
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["split_stratify"],
        random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["split_stratify"],
        random_state=42
    )
    
    print(f"Train size: {train_df.shape[0]}, Val size: {val_df.shape[0]}, Test size: {test_df.shape[0]}")
    
    # 4. Scale numerical columns
    print("Normalizing numerical inputs...")
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_df[numerical_features])
    X_val_num = scaler.transform(val_df[numerical_features])
    X_test_num = scaler.transform(test_df[numerical_features])
    
    # Input arrays
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
    
    # 5. Compute Class Weights & Sample Weights
    print("Computing class weights and sample weights...")
    unique_train_crops = np.unique(y_train_crop)
    crop_w = compute_class_weight(
        class_weight="balanced",
        classes=unique_train_crops,
        y=y_train_crop
    )
    crop_weights_dict = {c: float(w) for c, w in zip(unique_train_crops, crop_w)}
    for c in range(len(crop_encoder.classes_)):
        if c not in crop_weights_dict:
            crop_weights_dict[c] = 1.0
            
    unique_train_ferts = np.unique(y_train_fert)
    fert_w = compute_class_weight(
        class_weight="balanced",
        classes=unique_train_ferts,
        y=y_train_fert
    )
    fert_weights_dict = {f: float(w) for f, w in zip(unique_train_ferts, fert_w)}
    for f in range(len(fertilizer_encoder.classes_)):
        if f not in fert_weights_dict:
            fert_weights_dict[f] = 1.0
            
    # Generate sample weights arrays for multi-output training
    train_sample_weights_crop = np.array([crop_weights_dict[c] for c in y_train_crop])
    train_sample_weights_fert = np.array([fert_weights_dict[f] for f in y_train_fert])
            
    # 6. Build Keras Functional Model
    num_input = tf.keras.Input(shape=(6,), name="numerical")
    dist_input = tf.keras.Input(shape=(1,), name="district")
    soil_input = tf.keras.Input(shape=(1,), name="soil")
    
    # Embedding layers
    dist_embed = tf.keras.layers.Embedding(
        input_dim=len(district_encoder.classes_),
        output_dim=4
    )(dist_input)
    dist_embed = tf.keras.layers.Flatten()(dist_embed)
    
    soil_embed = tf.keras.layers.Embedding(
        input_dim=len(soil_encoder.classes_),
        output_dim=4
    )(soil_input)
    soil_embed = tf.keras.layers.Flatten()(soil_embed)
    
    # Concatenate numerical + categorical embeddings
    x = tf.keras.layers.Concatenate()([num_input, dist_embed, soil_embed])
    
    # Shared hidden layers
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    
    # Branch Outputs
    crop_output = tf.keras.layers.Dense(
        len(crop_encoder.classes_),
        activation="softmax",
        name="crop"
    )(x)
    
    fertilizer_output = tf.keras.layers.Dense(
        len(fertilizer_encoder.classes_),
        activation="softmax",
        name="fertilizer"
    )(x)
    
    model = tf.keras.Model(
        inputs=[num_input, dist_input, soil_input],
        outputs=[crop_output, fertilizer_output]
    )
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=1e-4),
        loss=[
            "sparse_categorical_crossentropy",
            "sparse_categorical_crossentropy"
        ],
        metrics=[
            ["accuracy"],
            ["accuracy"]
        ]
    )
    
    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    # 7. Fit Model
    print("\nStarting TensorFlow model training...")
    history = model.fit(
        x={
            "numerical": X_train_num,
            "district": X_train_dist,
            "soil": X_train_soil
        },
        y=[
            y_train_crop,
            y_train_fert
        ],
        validation_data=(
            {
                "numerical": X_val_num,
                "district": X_val_dist,
                "soil": X_val_soil
            },
            [
                y_val_crop,
                y_val_fert
            ]
        ),
        epochs=500,
        batch_size=64,
        sample_weight=[
            train_sample_weights_crop,
            train_sample_weights_fert
        ],
        callbacks=[early_stop, reduce_lr]
    )
    
    # 8. Evaluate on test set
    print("\nEvaluating model performance on test set...")
    predictions = model.predict({
        "numerical": X_test_num,
        "district": X_test_dist,
        "soil": X_test_soil
    })
    
    pred_crop_probs, pred_fert_probs = predictions
    y_pred_crop = np.argmax(pred_crop_probs, axis=1)
    y_pred_fert = np.argmax(pred_fert_probs, axis=1)
    
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
    
    # 9. Save Model and Metadata
    print("\nSaving final model checkpoint to saved_models_2/...")
    os.makedirs("saved_models_2", exist_ok=True)
    
    base_save_path = "saved_models_2/multitask_model"
    model_save_path = f"{base_save_path}.keras"
    meta_save_path = f"{base_save_path}_metadata.json"
    
    if os.path.exists(model_save_path):
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_save_path = f"{base_save_path}_{timestamp}.keras"
        meta_save_path = f"{base_save_path}_metadata_{timestamp}.json"
    
    # Save the Keras model
    model.save(model_save_path)
    
    # Save the scaler, unique mapping classes, and configs in a JSON metadata file
    metadata = {
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
    
    with open(meta_save_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Model successfully saved in {model_save_path}")
    print(f"Metadata successfully saved in {meta_save_path}")

if __name__ == "__main__":
    main()
