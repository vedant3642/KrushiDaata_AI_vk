"""
services/explainability_service.py

Model-agnostic SHAP feature-attribution layer.

Works against ANY predict_fn(X) -> (n_samples, n_crops) probability matrix,
so the SAME service explains the PyTorch regional model, the Keras regional
model, or the clubbed dual-stream ensemble -- whichever is active -- without
needing model-specific code. See app.py's `make_predict_fn()` for the
adapter that turns each model type into this uniform signature.
"""

import logging
import os
from typing import Callable, Dict, List, Any, Optional

import numpy as np
import shap

logger = logging.getLogger(__name__)


class ExplainabilityService:
    def __init__(
        self,
        feature_names: List[str],
        crop_names: List[str],
        background_data: np.ndarray,
    ):
        """
        feature_names: column order predict_fn expects, e.g.
            ["N", "P", "K", "ph", "rainfall", "temperature", "humidity"]
        crop_names: full ordered list of crop labels predict_fn returns
            probabilities for (must match the model's output order).
        background_data: (n_background, n_features) array used as SHAP's
            reference distribution. 50-100 rows sampled from real training
            data is far better than synthetic rows -- see
            `build_background_from_csv()` below.
        """
        if background_data.shape[1] != len(feature_names):
            raise ValueError(
                f"background_data has {background_data.shape[1]} columns but "
                f"{len(feature_names)} feature_names were given."
            )
        self.feature_names = feature_names
        self.crop_names = crop_names
        self.background_data = background_data
        # Cache one KernelExplainer per predict_fn (keyed by id) so repeated
        # calls for the same active model don't rebuild it every time.
        self._explainer_cache: Dict[int, "shap.KernelExplainer"] = {}
        # Kernel SHAP returns attributions for every crop class in one call.
        # Retain that result for this input so explaining the winner and then
        # comparing alternatives does not re-run the same expensive sampling.
        self._shap_values_cache: Dict[tuple, Any] = {}

    def _get_explainer(self, predict_fn: Callable[[np.ndarray], np.ndarray]) -> "shap.KernelExplainer":
        key = id(predict_fn)
        if key not in self._explainer_cache:
            self._explainer_cache[key] = shap.KernelExplainer(predict_fn, self.background_data)
        return self._explainer_cache[key]

    def explain(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        feature_vector: np.ndarray,
        target_crop: str,
        nsamples: int = 24,
    ) -> Dict[str, Any]:
        """
        Returns per-feature SHAP contribution toward `target_crop`'s predicted
        probability for this single instance, ranked by magnitude, plus a
        natural-language summary suitable for direct display to a farmer.
        """
        if target_crop not in self.crop_names:
            raise ValueError(f"Unknown crop '{target_crop}'")

        explainer = self._get_explainer(predict_fn)
        x = feature_vector.reshape(1, -1).astype(float)
        crop_idx = self.crop_names.index(target_crop)

        cache_key = (id(predict_fn), tuple(x.ravel()), nsamples)
        if cache_key not in self._shap_values_cache:
            self._shap_values_cache[cache_key] = explainer.shap_values(
                x, nsamples=nsamples
            )
        raw_shap = self._shap_values_cache[cache_key]
        values = self._extract_class_values(raw_shap, crop_idx)

        total_abs = float(np.sum(np.abs(values))) or 1e-9
        contributions = []
        for fname, val in zip(self.feature_names, values):
            contributions.append({
                "feature": fname,
                "shap_value": float(val),
                "contribution_pct": float(val / total_abs * 100.0),
                "direction": "increased" if val > 0 else "decreased",
            })
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

        return {
            "target_crop": target_crop,
            "contributions": contributions,
            "natural_language_summary": self._to_natural_language(target_crop, contributions),
        }

    def explain_rejection(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        feature_vector: np.ndarray,
        chosen_crop: str,
        rejected_crop: str,
        nsamples: int = 24,
    ) -> Dict[str, Any]:
        """
        "Why not X?" -- compares each feature's SHAP contribution toward the
        chosen crop vs. the rejected crop, and surfaces the features where the
        gap is largest (i.e. the features that most tipped the decision away
        from the rejected crop).
        """
        chosen = self.explain(predict_fn, feature_vector, chosen_crop, nsamples)
        rejected = self.explain(predict_fn, feature_vector, rejected_crop, nsamples)

        chosen_map = {c["feature"]: c["shap_value"] for c in chosen["contributions"]}
        rejected_map = {c["feature"]: c["shap_value"] for c in rejected["contributions"]}

        gaps = []
        for fname in self.feature_names:
            gap = chosen_map[fname] - rejected_map[fname]
            gaps.append({
                "feature": fname,
                "favored_chosen_by": float(gap),
                "chosen_shap": chosen_map[fname],
                "rejected_shap": rejected_map[fname],
            })
        gaps.sort(key=lambda g: abs(g["favored_chosen_by"]), reverse=True)

        top = gaps[:3]
        reasons = [f"{g['feature']} favored {chosen_crop} over {rejected_crop}" for g in top]
        summary = f"{rejected_crop} was not selected mainly because " + "; ".join(reasons) + "."

        return {
            "chosen_crop": chosen_crop,
            "rejected_crop": rejected_crop,
            "feature_gaps": gaps,
            "natural_language_summary": summary,
        }

    @staticmethod
    def _extract_class_values(raw_shap: Any, crop_idx: int) -> np.ndarray:
        """
        Normalizes SHAP's multi-output return shape across library versions:
        older shap returns a list of (n_samples, n_features) arrays (one per
        class); newer shap returns a single (n_samples, n_features, n_classes)
        array. Both are reduced to a flat (n_features,) vector for one sample.
        """
        if isinstance(raw_shap, list):
            return np.array(raw_shap[crop_idx][0])
        arr = np.array(raw_shap)
        if arr.ndim == 3:
            return arr[0, :, crop_idx]
        return arr[0]  # single-output model fallback

    @staticmethod
    def _to_natural_language(crop: str, contributions: List[Dict[str, Any]], top_n: int = 3) -> str:
        top = contributions[:top_n]
        parts = []
        for c in top:
            verb = "favored" if c["shap_value"] > 0 else "worked against"
            parts.append(f"{c['feature']} {verb} {crop} ({c['contribution_pct']:+.1f}%)")
        return f"{crop} was recommended primarily because " + ", ".join(parts) + "."


def build_background_from_csv(
    csv_paths: Any,
    feature_columns: List[str],
    sample_size: int = 75,
    random_state: int = 42,
) -> np.ndarray:
    """
    Convenience helper: samples `sample_size` real rows from one or more
    training CSVs (e.g. Crop_recommendation.csv and regional CSV) as the SHAP
    background distribution.
    """
    import pandas as pd
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    
    dfs = []
    col_mapping = {
        "nitrogen": "N",
        "phosphorus": "P",
        "potassium": "K",
        "ph": "ph",
        "rainfall": "rainfall",
        "temperature": "temperature",
        "humidity": "humidity"
    }

    for path in csv_paths:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        # normalize column names
        norm_cols = {c: col_mapping.get(c.strip().lower(), c.strip()) for c in df.columns}
        df = df.rename(columns=norm_cols)
        
        # fill missing feature columns with standard defaults if any (e.g. humidity in regional csv)
        for col in feature_columns:
            if col not in df.columns:
                if col.lower() == "humidity":
                    df[col] = 70.0
                else:
                    df[col] = 0.0
        
        dfs.append(df[feature_columns])

    if not dfs:
        raise ValueError(f"No valid CSV files found in paths: {csv_paths}")

    combined = pd.concat(dfs, ignore_index=True)
    sample = combined.sample(
        n=min(sample_size, len(combined)), random_state=random_state
    )
    return sample.to_numpy(dtype=float)


_explainability_service_instance: Optional[ExplainabilityService] = None


def get_explainability_service(
    feature_names: List[str],
    crop_names: List[str],
    background_data: np.ndarray,
) -> ExplainabilityService:
    global _explainability_service_instance
    # A Streamlit session can switch between checkpoints with different crop
    # output orders. Never reuse an explainer configured for another order.
    if (
        _explainability_service_instance is None
        or _explainability_service_instance.feature_names != feature_names
        or _explainability_service_instance.crop_names != crop_names
    ):
        _explainability_service_instance = ExplainabilityService(
            feature_names, crop_names, background_data
        )
    return _explainability_service_instance
