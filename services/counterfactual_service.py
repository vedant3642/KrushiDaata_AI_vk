"""
services/counterfactual_service.py

Model-agnostic counterfactual reasoning engine.

Given a rejected crop, finds the smallest realistic change to ACTIONABLE
soil/water features (N, P, K, pH, rainfall) that would make it the top
prediction -- while holding non-actionable climate features (temperature,
humidity) fixed. This mirrors the mutability-mask approach from DiCE-style
counterfactual recourse (Wachter et al., 2017): climate is context you
can't change, soil chemistry and irrigation are decisions a farmer can act on.

Works against the SAME predict_fn(X) -> probability_matrix signature used
by ExplainabilityService, so it is equally model-agnostic.
"""

import logging
from typing import Callable, Dict, List, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Realistic agronomic bounds. Tune these against your own training data's
# actual min/max (e.g. from metadata.json) rather than trusting these as-is --
# they're reasonable Maharashtra-context defaults, not universal truths.
DEFAULT_FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "N": (0.0, 140.0),
    "P": (0.0, 145.0),
    "K": (0.0, 205.0),
    "ph": (3.5, 9.5),
    "rainfall": (300.0, 2500.0),
    "temperature": (8.0, 45.0),
    "humidity": (10.0, 100.0),
}

# Only these are searched over; temperature/humidity stay fixed at whatever
# the farmer's actual conditions are (a farmer cannot change the weather).
DEFAULT_ACTIONABLE_FEATURES: List[str] = ["N", "P", "K", "ph", "rainfall"]


class CounterfactualService:
    def __init__(
        self,
        feature_names: List[str],
        crop_names: List[str],
        feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        actionable_features: Optional[List[str]] = None,
        step_fraction: float = 0.05,
    ):
        self.feature_names = feature_names
        self.crop_names = crop_names
        self.feature_bounds = feature_bounds or DEFAULT_FEATURE_BOUNDS
        self.actionable_features = actionable_features or [
            f for f in DEFAULT_ACTIONABLE_FEATURES if f in feature_names
        ]
        self.step_fraction = step_fraction

        missing_bounds = [f for f in feature_names if f not in self.feature_bounds]
        if missing_bounds:
            raise ValueError(f"No bounds defined for features: {missing_bounds}")

    def find_counterfactual(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        current_features: np.ndarray,
        target_crop: str,
        max_iterations: int = 24,
    ) -> Dict[str, Any]:
        """
        Bounded greedy hill-climb: at each step, tries a small step in every
        actionable feature (both directions), keeps whichever single step
        increases target_crop's probability the most, and repeats until
        target_crop becomes the top prediction, no step helps further, or
        max_iterations is hit.
        """
        if target_crop not in self.crop_names:
            raise ValueError(f"Unknown crop '{target_crop}'")

        target_idx = self.crop_names.index(target_crop)
        x = current_features.copy().astype(float)

        # Predict every possible one-step change as one batch.  The former
        # implementation performed separate model calls for each direction,
        # feature and bookkeeping check (over 700 calls per alternative).
        # This keeps the same greedy search but needs at most 25 calls.
        current_probs = predict_fn(x.reshape(1, -1))[0]
        starting_prob = float(current_probs[target_idx])
        steps_taken = []

        for _ in range(max_iterations):
            if self.crop_names[int(np.argmax(current_probs))] == target_crop:
                break

            trials = []
            trial_changes = []

            for fname in self.actionable_features:
                f_idx = self.feature_names.index(fname)
                lo, hi = self.feature_bounds[fname]
                step = (hi - lo) * self.step_fraction

                for direction in (+1, -1):
                    trial = x.copy()
                    trial[f_idx] = float(np.clip(trial[f_idx] + direction * step, lo, hi))
                    if trial[f_idx] == x[f_idx]:
                        continue  # already pinned at a bound
                    trials.append(trial)
                    trial_changes.append((fname, f_idx, trial[f_idx]))

            if not trials:
                break  # stuck: no further single step improves the target crop

            trial_probs = predict_fn(np.asarray(trials, dtype=float))
            gains = trial_probs[:, target_idx] - current_probs[target_idx]
            best_trial_idx = int(np.argmax(gains))
            if float(gains[best_trial_idx]) <= 0.0:
                break  # no realistic single change improves the target crop

            best_feature, f_idx, best_new_val = trial_changes[best_trial_idx]
            x[f_idx] = best_new_val
            current_probs = trial_probs[best_trial_idx]
            steps_taken.append({"feature": best_feature, "new_value": float(best_new_val)})

        achieved = self.crop_names[int(np.argmax(current_probs))] == target_crop
        final_prob = float(current_probs[target_idx])

        deltas = []
        for fname in self.actionable_features:
            f_idx = self.feature_names.index(fname)
            delta = x[f_idx] - current_features[f_idx]
            if abs(delta) > 1e-6:
                lo, hi = self.feature_bounds[fname]
                deltas.append({
                    "feature": fname,
                    "current_value": float(current_features[f_idx]),
                    "required_value": round(float(x[f_idx]), 2),
                    "delta": round(float(delta), 2),
                    "direction": "increase" if delta > 0 else "decrease",
                    "pct_of_realistic_range": float(abs(delta) / (hi - lo) * 100.0),
                })

        feasibility_score = self._feasibility_score(deltas)

        return {
            "target_crop": target_crop,
            "achieved": achieved,
            "iterations_used": len(steps_taken),
            "starting_target_probability_pct": round(starting_prob * 100, 2),
            "final_target_probability_pct": round(final_prob * 100, 2),
            "deltas": deltas,
            "feasibility_score": feasibility_score,
            "feasibility_label": self._feasibility_label(feasibility_score, achieved),
        }

    @staticmethod
    def _feasibility_score(deltas: List[Dict[str, Any]]) -> float:
        """
        0-100: 100 = trivial (tiny change), 0 = would require exhausting most
        of the realistic range across multiple features simultaneously.
        """
        if not deltas:
            return 100.0
        avg_pct_of_range = float(np.mean([d["pct_of_realistic_range"] for d in deltas]))
        return round(max(0.0, 100.0 - avg_pct_of_range * 1.5), 1)

    @staticmethod
    def _feasibility_label(score: float, achieved: bool) -> str:
        if not achieved:
            return "Not reachable within realistic agronomic bounds"
        if score >= 70:
            return "Easily achievable"
        if score >= 40:
            return "Moderately achievable"
        return "Difficult — requires substantial intervention"


_counterfactual_service_instance: Optional[CounterfactualService] = None


def get_counterfactual_service(
    feature_names: List[str],
    crop_names: List[str],
    feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    actionable_features: Optional[List[str]] = None,
) -> CounterfactualService:
    global _counterfactual_service_instance
    # Do not keep a crop-index mapping from a previously selected checkpoint.
    if (
        _counterfactual_service_instance is None
        or _counterfactual_service_instance.feature_names != feature_names
        or _counterfactual_service_instance.crop_names != crop_names
    ):
        _counterfactual_service_instance = CounterfactualService(
            feature_names, crop_names, feature_bounds, actionable_features
        )
    return _counterfactual_service_instance
