# Project Guide & AI Agent Context (CLAUDE.md)

## System Architecture & Model Selection Rationale

### ⚠️ Known Issue: Rainfall Unit Mismatch (RESOLVED)
- The weather widget shows **7-Day forecasted rain in mm** (e.g., 16.6 mm).
- The model expects **Annual Rainfall in mm/year** (300–1700 mm for Maharashtra).
- **Fix implemented in `app.py`:**
  - Warning banner inside the weather expander explaining the mismatch.
  - Rainfall input field uses `st.session_state["rainfall_val"]` and `st.session_state["rainfall_source"]`.
  - **"📊 Calculate Annual Rainfall (Last 12 Months)"** button calls `weather_service.fetch_annual_rainfall(district)` which queries the Open-Meteo Archive API (`archive-api.open-meteo.com/v1/archive`) for the past 365 days of `precipitation_sum` and auto-fills the correct annual figure.
  - If auto-filled, the label shows ✅ indicator; if manually edited, it reverts to "manual" state.

### Why Ensemble (Clubbed Model) vs. Single Regional Model?

Although evaluated on the regional test set alone, both the single regional model (`multitask_model0.keras`) and the dual-stream clubbed ensemble (`saved_models_clubbed`) produce identical metrics (**99.85% Crop Accuracy**, **77.84% Fertilizer Accuracy**), there are critical domain and architectural reasons why dual-stream ensembling is designed:

1. **Domain & Feature Coverage Complementarity**:
   - **Regional Model (`multitask_model0.keras`)**: Trained on district-specific Maharashtra agricultural data (`Crop and fertilizer dataset (1).csv`). Highly specialized in district/soil context (Kolhapur, Pune, Sangli, Satara, Solapur), but only aware of 16 regional crop labels and lacks granular relative humidity dynamics.
   - **Agronomic Model (`agronomic_crop_model.pt`)**: Trained on pure agronomic biological boundaries across 22 crops (`Crop_recommendation.csv`) including humidity thresholds.
   - **Ensemble Fusion**: Fuses 72% regional weight (district/soil context authority) with 28% agronomic weight (fine-grained biological suitability validation).

2. **Out-of-Distribution (OOD) & Cross-Dataset Reliability**:
   - Single models risk overfitting to specific regional survey feature distributions.
   - Dual-stream weighted consensus prevents catastrophic misclassification when inputs lie near decision boundaries or when novel weather patterns emerge.

3. **Trade-off Analysis (Single vs Clubbed)**:
   - **If Low Latency is Top Priority (< 1ms)**: Deploy `multitask_model0.keras` directly.
   - **If Maximum Generalization & Biological Validation are Top Priority**: Deploy `ClubbedEnsemblePredictor` (`saved_models_clubbed`).

---

## Directory Structure & Key Models

- `saved_models_2/`:
  - `multitask_model0.keras`: Best regional Multi-Task Keras model trained on augmented data (Crop Acc: 99.85%, Fert Acc: 77.84%).
  - `multitask_model.pt`: PyTorch Multi-Task baseline model.
  - `multitask_model.keras`: Keras baseline model.
  - `multitask_model_real.keras`: Model trained on non-augmented real data only.

- `saved_models_clubbed/`:
  - `multitask_model0.keras` / `regional_multitask_model.pt`: Regional stream.
  - `agronomic_crop_model.pt`: Standalone PyTorch agronomic crop classifier (22 crops, 99.05% test acc).
  - `ensemble_metadata.json`: Cross-dataset mapping (e.g., Gram -> Chickpea, Tur -> Pigeonpeas) and weight configs.

- `services/`:
  - `clubbed_prediction_service.py`: Implements `ClubbedEnsemblePredictor`. Automatically detects Keras/PyTorch models and runs dual-stream weighted fusion.
  - `regional_stats_service.py`: Historical area, production, and yield statistics service.
  - `weather_service.py`: Live weather data integration.

---

## Key Verification & Evaluation Scripts

- `evaluate_all_models_detailed.py`: Runs comprehensive benchmark metrics (Accuracy, Macro F1, Latency) across all models in all folders.
- `compare_all_models.py`: Summary evaluation script comparing PyTorch, Keras, and Clubbed models side-by-side.
- `app.py`: Streamlit web application user interface.

---

## Command Reference

When executing scripts in this repository, always use the virtual environment Python interpreter:

```powershell
.\venv\Scripts\python.exe evaluate_all_models_detailed.py
.\venv\Scripts\python.exe compare_all_models.py
.\venv\Scripts\python.exe -m streamlit run app.py
```
