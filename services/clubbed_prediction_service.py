import os
import json
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# Recreate model architectures for inference
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


class RegionalMultiTaskModel(nn.Module):
    def __init__(self, num_districts, num_soils, num_numerical, num_crops, num_fertilizers):
        super().__init__()
        self.district_embed = nn.Embedding(num_districts, 4)
        self.soil_embed = nn.Embedding(num_soils, 4)

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

        self.crop_head = nn.Linear(64, num_crops)
        self.fert_head = nn.Linear(64, num_fertilizers)

    def forward(self, x_num, x_dist, x_soil):
        dist_emb = self.district_embed(x_dist)
        soil_emb = self.soil_embed(x_soil)
        x = torch.cat([x_num, dist_emb, soil_emb], dim=1)
        shared_out = self.shared(x)
        crop_logits = self.crop_head(shared_out)
        fert_logits = self.fert_head(shared_out)
        return crop_logits, fert_logits


class ClubbedEnsemblePredictor:
    """
    Dual-stream ensemble that fuses:
      - Regional model (district-aware, 72% weight by default)
      - Agronomic model (biological/22-crop, 28% weight by default)

    IMPORTANT: fusion only happens for crops that exist in BOTH datasets
    (via the regional_to_agronomic name mapping). Crops absent from that
    mapping silently fall back to regional-only scoring. This class now
    tracks and exposes that per-crop, so the UI can show it instead of
    hiding it.
    """

    def __init__(self, model_dir="saved_models_clubbed", district_weight=0.72):
        self.model_dir = model_dir
        self.district_weight = district_weight
        self.agronomic_weight = 1.0 - district_weight

        self._load_models()

    def _load_models(self):
        # 1. Check & Load Regional Multi-Task Model (supports .keras, .h5, and .pt)
        keras_candidates = [
            os.path.join(self.model_dir, "multitask_model0.keras"),
            os.path.join(self.model_dir, "regional_multitask_model.keras"),
            os.path.join(self.model_dir, "multitask_model.keras")
        ]
        torch_candidates = [
            os.path.join(self.model_dir, "regional_multitask_model.pt"),
            os.path.join(self.model_dir, "multitask_model.pt")
        ]

        reg_keras_path = next((p for p in keras_candidates if os.path.exists(p)), None)
        reg_torch_path = next((p for p in torch_candidates if os.path.exists(p)), None)

        # Load mappings and scaler metadata
        meta_path = os.path.join(self.model_dir, "ensemble_metadata.json")
        standalone_meta_0 = os.path.join(self.model_dir, "multitask_model_metadata0.json")
        standalone_meta = os.path.join(self.model_dir, "multitask_model_metadata.json")

        # Track where the cross-dataset mapping came from, for diagnostics
        self.mapping_source = None

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                ensemble_meta = json.load(f)
            reg_meta = ensemble_meta.get("regional_model", {})
            self.reg_districts = reg_meta.get("districts", [])
            self.reg_soils = reg_meta.get("soils", [])
            self.reg_crops = reg_meta.get("crops", [])
            self.reg_fertilizers = reg_meta.get("fertilizers", [])
            self.reg_scaler_mean = np.array(reg_meta.get("scaler", {}).get("mean", []))
            self.reg_scaler_scale = np.array(reg_meta.get("scaler", {}).get("scale", []))
            # Normalize only valid cross-dataset mappings.  A null value means
            # that the regional crop has no agronomic counterpart and should
            # use the documented regional-only fallback; it must not prevent
            # the complete ensemble from being offered in the sidebar.
            raw_map = ensemble_meta.get("mappings", {}).get("regional_to_agronomic", {})
            self.reg_to_agro = {
                str(regional_crop).lower(): str(agronomic_crop).lower()
                for regional_crop, agronomic_crop in raw_map.items()
                if regional_crop and agronomic_crop
            }
            self.mapping_source = "ensemble_metadata.json"
        elif os.path.exists(standalone_meta_0) or os.path.exists(standalone_meta):
            m_path = standalone_meta_0 if os.path.exists(standalone_meta_0) else standalone_meta
            with open(m_path, "r") as f:
                reg_meta = json.load(f)
            self.reg_districts = reg_meta["mappings"]["districts"]
            self.reg_soils = reg_meta["mappings"]["soils"]
            self.reg_crops = reg_meta["mappings"]["crops"]
            self.reg_fertilizers = reg_meta["mappings"]["fertilizers"]
            self.reg_scaler_mean = np.array(reg_meta["scaler"]["mean"])
            self.reg_scaler_scale = np.array(reg_meta["scaler"]["scale"])
            self.reg_to_agro = {}
            self.mapping_source = "none (standalone metadata only)"
        else:
            raise FileNotFoundError(f"Could not find metadata JSON in {self.model_dir}")

        # Surface the degraded-mode condition loudly instead of silently proceeding
        self.mapping_loaded = bool(self.reg_to_agro)
        if not self.mapping_loaded:
            logger.warning(
                "ClubbedEnsemblePredictor: no regional\u2192agronomic crop name mapping was "
                "loaded (source: %s). Every crop will fall back to regional-only scoring "
                "unless its name matches the agronomic dataset's class name exactly.",
                self.mapping_source,
            )

        if reg_keras_path:
            import tensorflow as tf
            self.is_keras_regional = True
            self.reg_model = tf.keras.models.load_model(reg_keras_path)
            self.reg_model_path = reg_keras_path
        elif reg_torch_path:
            self.is_keras_regional = False
            reg_ckpt = torch.load(reg_torch_path, map_location="cpu", weights_only=False)
            cfg = reg_ckpt["model_config"]
            self.reg_model = RegionalMultiTaskModel(
                num_districts=cfg["num_districts"],
                num_soils=cfg["num_soils"],
                num_numerical=cfg["num_numerical"],
                num_crops=cfg["num_crops"],
                num_fertilizers=cfg["num_fertilizers"]
            )
            self.reg_model.load_state_dict(reg_ckpt["model_state_dict"])
            self.reg_model.eval()
            self.reg_model_path = reg_torch_path
        else:
            raise FileNotFoundError(f"No regional multi-task model found in {self.model_dir}")

        # 2. Load Agronomic Crop Model (PyTorch)
        agro_path = os.path.join(self.model_dir, "agronomic_crop_model.pt")
        if not os.path.exists(agro_path):
            raise FileNotFoundError(f"Agronomic crop model not found at {agro_path}")
        agro_ckpt = torch.load(agro_path, map_location="cpu", weights_only=False)
        agro_cfg = agro_ckpt["model_config"]

        self.agro_model = AgronomicCropModel(
            num_features=agro_cfg["num_features"],
            num_classes=agro_cfg["num_classes"]
        )
        self.agro_model.load_state_dict(agro_ckpt["model_state_dict"])
        self.agro_model.eval()

        # Normalize agronomic class names to lowercase too, so lookups are case-insensitive
        self.agro_crops = [c.lower() for c in agro_ckpt["classes"]]
        self.agro_scaler_mean = np.array(agro_ckpt["scaler"]["mean"])
        self.agro_scaler_scale = np.array(agro_ckpt["scaler"]["scale"])

    def predict(
        self,
        district: str,
        soil: str,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        ph: float,
        rainfall: float,
        temperature: float,
        humidity: float = 65.0,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Executes dual-stream ensemble inference:
        - Prioritizes the Regional District model (72% weight) for proven local compatibility.
        - Fuses with the Agronomic model (28% weight) for fine-grained biological thresholding.

        Every returned crop now carries a "fusion_status" field:
          - "dual_stream"        -> both models contributed (true ensemble)
          - "regional_only"      -> crop had no agronomic-dataset counterpart; regional model
                                     had full authority for this crop specifically
        """
        # Safe indexing for categoricals
        dist_idx = self.reg_districts.index(district) if district in self.reg_districts else 0
        soil_idx = self.reg_soils.index(soil) if soil in self.reg_soils else 0

        # 1. Regional Model Inference
        # Features: [N, P, K, pH, Rainfall, Temperature]
        reg_num_raw = np.array([nitrogen, phosphorus, potassium, ph, rainfall, temperature], dtype=np.float32)
        reg_num_scaled = (reg_num_raw - self.reg_scaler_mean) / self.reg_scaler_scale

        if self.is_keras_regional:
            import tensorflow as tf
            reg_preds = self.reg_model({
                "numerical": tf.convert_to_tensor(reg_num_scaled.reshape(1, -1), dtype=tf.float32),
                "district": tf.convert_to_tensor([dist_idx], dtype=tf.int32),
                "soil": tf.convert_to_tensor([soil_idx], dtype=tf.int32)
            }, training=False)
            reg_crop_probs = reg_preds[0].numpy()[0]
            reg_fert_probs = reg_preds[1].numpy()[0]
        else:
            t_num = torch.FloatTensor(reg_num_scaled).unsqueeze(0)
            t_dist = torch.LongTensor([dist_idx])
            t_soil = torch.LongTensor([soil_idx])
            with torch.no_grad():
                reg_crop_logits, reg_fert_logits = self.reg_model(t_num, t_dist, t_soil)
                reg_crop_probs = torch.softmax(reg_crop_logits, dim=1).squeeze(0).numpy()
                reg_fert_probs = torch.softmax(reg_fert_logits, dim=1).squeeze(0).numpy()

        # 2. Agronomic Model Inference
        # Features: [N, P, K, Temperature, Humidity, pH, Rainfall]
        agro_num_raw = np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall], dtype=np.float32)
        agro_num_scaled = (agro_num_raw - self.agro_scaler_mean) / self.agro_scaler_scale

        t_agro_num = torch.FloatTensor(agro_num_scaled).unsqueeze(0)
        with torch.no_grad():
            agro_logits = self.agro_model(t_agro_num)
            agro_probs = torch.softmax(agro_logits, dim=1).squeeze(0).numpy()

        # 3. Weighted Logit & Probability Fusion (Weighted Consensus)
        fused_crop_scores = {}
        fusion_status = {}
        for idx, crop in enumerate(self.reg_crops):
            reg_prob = float(reg_crop_probs[idx])

            # Check if this crop exists in agronomic model
            crop_lower = crop.lower()
            mapped_agro_crop = self.reg_to_agro.get(crop_lower, crop_lower)

            if mapped_agro_crop in self.agro_crops:
                agro_idx = self.agro_crops.index(mapped_agro_crop)
                agro_prob = float(agro_probs[agro_idx])
                fusion_status[crop] = "dual_stream"
            else:
                # If regional crop is unique to regional data (e.g., Sugarcane, Soybean, Turmeric)
                # the regional model has full authority -- this is a real fallback, not fusion.
                agro_prob = reg_prob
                fusion_status[crop] = "regional_only"

            # Weighted ensemble combination (72% district-regional authority)
            combined_score = (self.district_weight * reg_prob) + (self.agronomic_weight * agro_prob)
            fused_crop_scores[crop] = combined_score

        # Normalize fused scores
        total_score = sum(fused_crop_scores.values())
        if total_score > 0:
            fused_crop_scores = {c: s / total_score for c, s in fused_crop_scores.items()}

        # Sort top crops
        ranked_crops = sorted(fused_crop_scores.items(), key=lambda x: x[1], reverse=True)
        top_crops = [
            {
                "crop": c,
                "confidence": round(p * 100, 2),
                "fusion_status": fusion_status[c],
            }
            for c, p in ranked_crops[:top_k]
        ]

        # Best Fertilizer
        top_fert_idx = np.argmax(reg_fert_probs)
        top_fert = self.reg_fertilizers[top_fert_idx]
        top_fert_conf = float(reg_fert_probs[top_fert_idx])

        # Summary diagnostics: how many of the returned crops actually got dual-stream fusion
        dual_count = sum(1 for c in top_crops if c["fusion_status"] == "dual_stream")

        return {
            "recommended_crop": top_crops[0]["crop"],
            "crop_confidence": top_crops[0]["confidence"],
            "top_crops": top_crops,
            "recommended_fertilizer": top_fert,
            "fertilizer_confidence": round(top_fert_conf * 100, 2),
            "district_weight_used": self.district_weight,
            "agronomic_weight_used": self.agronomic_weight,
            "mapping_loaded": self.mapping_loaded,
            "mapping_source": self.mapping_source,
            "dual_stream_count": dual_count,
            "total_returned": len(top_crops),
        }
