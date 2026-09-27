import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from pathlib import Path
from dotenv import load_dotenv

# Add workspace directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(Path(__file__).with_name(".env"))
from services.regional_stats_service import get_regional_stats_service, CROP_TYPES_MAP
from services.weather_service import get_weather_service
from services.clubbed_prediction_service import ClubbedEnsemblePredictor
from services.explainability_service import get_explainability_service, build_background_from_csv
from services.counterfactual_service import get_counterfactual_service
from services.chatbot_service import get_chatbot_service


# This is the shared column order used by the XAI services.  The regional
# models consume the first six fields; the ensemble additionally consumes
# humidity through its agronomic stream.
XAI_FEATURE_ORDER = ["N", "P", "K", "ph", "rainfall", "temperature", "humidity"]


def make_predict_fn(active_model_type, active_model_obj, dist_idx, soil_idx,
                    scaler_mean, scaler_scale, crops):
    """Adapt the active model to ``predict_fn(X) -> crop probabilities``.

    ``X`` must use ``XAI_FEATURE_ORDER`` and district/soil remain fixed to
    the farmer's selected context while the explanation varies its features.
    """
    if active_model_type == "clubbed_ensemble":
        def predict_fn(X):
            out = np.zeros((X.shape[0], len(crops)))
            district = active_model_obj.reg_districts[dist_idx]
            soil = active_model_obj.reg_soils[soil_idx]
            for i, row in enumerate(X):
                N, P, K, ph, rainfall, temperature, humidity = row
                result = active_model_obj.predict(
                    district=district, soil=soil,
                    nitrogen=N, phosphorus=P, potassium=K, ph=ph,
                    rainfall=rainfall, temperature=temperature, humidity=humidity,
                    top_k=len(crops),
                )
                probability_by_crop = {
                    item["crop"]: item["confidence"] / 100.0
                    for item in result["top_crops"]
                }
                out[i] = [probability_by_crop.get(crop, 0.0) for crop in crops]
            return out
        return predict_fn

    if active_model_type == "pytorch":
        def predict_fn(X):
            scaled = (X[:, :6] - scaler_mean) / scaler_scale
            x_num = torch.FloatTensor(scaled)
            x_dist = torch.LongTensor([dist_idx] * X.shape[0])
            x_soil = torch.LongTensor([soil_idx] * X.shape[0])
            with torch.no_grad():
                crop_logits, _ = active_model_obj(x_num, x_dist, x_soil)
                return torch.softmax(crop_logits, dim=1).numpy()
        return predict_fn

    def predict_fn(X):
        scaled = (X[:, :6] - scaler_mean) / scaler_scale
        inputs = {
            "numerical": scaled.astype(np.float32),
            "district": np.full((X.shape[0], 1), dist_idx, dtype=np.int32),
            "soil": np.full((X.shape[0], 1), soil_idx, dtype=np.int32),
        }
        predictions = active_model_obj(inputs, training=False)
        return predictions[0].numpy()
    return predict_fn

# ==========================================
# 1. Model Definition (must match train scripts)
# ==========================================
class MultiTaskModel(nn.Module):
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

# ==========================================
# 2. Cache Resources
# ==========================================
@st.cache_resource
def load_regional_service():
    return get_regional_stats_service()

@st.cache_resource
def load_weather_service():
    return get_weather_service()

@st.cache_resource
def load_pytorch_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
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
    return model, checkpoint["mappings"], checkpoint["scaler"]

@st.cache_resource
def load_keras_model(model_path, metadata_path):
    import tensorflow as tf
    import json

    model = tf.keras.models.load_model(model_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return model, metadata["mappings"], metadata["scaler"]

@st.cache_resource
def load_clubbed_predictor(model_dir="saved_models_clubbed"):
    return ClubbedEnsemblePredictor(model_dir=str(model_dir), district_weight=0.72)


@st.cache_resource
def load_xai_services(crop_names):
    """Load shared SHAP background data and counterfactual configuration once."""
    background = build_background_from_csv(
        "Crop_recommendation.csv",
        feature_columns=XAI_FEATURE_ORDER,
        # Kernel SHAP scales with both this background size and nsamples.
        # Eight real training rows keep the interactive explanation responsive
        # for the slower dual-stream ensemble while retaining data grounding.
        sample_size=8,
    )
    explain_service = get_explainability_service(
        XAI_FEATURE_ORDER, list(crop_names), background
    )
    counterfactual_service = get_counterfactual_service(
        XAI_FEATURE_ORDER, list(crop_names),
    )
    return explain_service, counterfactual_service


@st.cache_resource
def load_chatbot():
    """Create one reusable Groq chatbot client using the .env API key."""
    return get_chatbot_service(
        provider="groq",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        # Selected from the models currently available to this Groq key.
        groq_model="openai/gpt-oss-20b",
    )


# Initialize Regional Stats Service & Weather Service
reg_service = load_regional_service()
weather_service = load_weather_service()
chatbot = load_chatbot()

# ==========================================
# 3. Streamlit Page Configuration
# ==========================================
st.set_page_config(
    page_title="KrushiDaata - Smart Crop & Fertilizer Recommendation",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 KrushiDaata: Smart Multi-Task Crop & Fertilizer System")
st.markdown("""
Empowering farmers with **Deep Learning Agronomic Recommendations** coupled with **Historical Regional Crop Yield, Risk Volatility, and Regional Plausibility Analytics**.
""")

# Dynamic Model Discovery in saved_models_2/ and saved_models_clubbed/
# Keep exactly three intended choices in the sidebar dropdown:
# 1) PyTorch regional model
# 2) Keras regional model using metadata0
# 3) Clubbed ensemble
saved_dir = Path("saved_models_2")
clubbed_dir = Path("saved_models_clubbed")
loaded_models_registry = {}

# 1. Discover Clubbed Ensemble. Accept either regional .pt or .keras files.
regional_candidates = [
    clubbed_dir / "regional_multitask_model.pt",
    clubbed_dir / "multitask_model.pt",
    clubbed_dir / "multitask_model0.keras",
    clubbed_dir / "regional_multitask_model.keras",
]
if clubbed_dir.exists() and (clubbed_dir / "agronomic_crop_model.pt").exists() and any(p.exists() for p in regional_candidates):
    try:
        predictor = load_clubbed_predictor(clubbed_dir)

        if (clubbed_dir / "ensemble_metadata.json").exists():
            with open(clubbed_dir / "ensemble_metadata.json", "r") as f:
                meta = json.load(f)
            reg_meta = meta.get("regional_model", {})
            mappings = {
                "districts": reg_meta.get("districts", predictor.reg_districts),
                "soils": reg_meta.get("soils", predictor.reg_soils),
                "crops": reg_meta.get("crops", predictor.reg_crops),
                "fertilizers": reg_meta.get("fertilizers", predictor.reg_fertilizers),
            }
            scaler = reg_meta.get("scaler", {
                "mean": predictor.reg_scaler_mean.tolist(),
                "scale": predictor.reg_scaler_scale.tolist(),
            })
        elif (clubbed_dir / "multitask_model_metadata0.json").exists():
            with open(clubbed_dir / "multitask_model_metadata0.json", "r") as f:
                meta = json.load(f)
            mappings = meta.get("mappings", {
                "districts": predictor.reg_districts,
                "soils": predictor.reg_soils,
                "crops": predictor.reg_crops,
                "fertilizers": predictor.reg_fertilizers,
            })
            scaler = meta.get("scaler", {
                "mean": predictor.reg_scaler_mean.tolist(),
                "scale": predictor.reg_scaler_scale.tolist(),
            })
        else:
            mappings = {
                "districts": predictor.reg_districts,
                "soils": predictor.reg_soils,
                "crops": predictor.reg_crops,
                "fertilizers": predictor.reg_fertilizers,
            }
            scaler = {
                "mean": predictor.reg_scaler_mean.tolist(),
                "scale": predictor.reg_scaler_scale.tolist(),
            }

        display_name = "Clubbed Ensemble Model"
        loaded_models_registry[display_name] = {
            "type": "clubbed_ensemble",
            "model": predictor,
            "mappings": mappings,
            "scaler": scaler,
            "file": "saved_models_clubbed"
        }
    except Exception as e:
        st.sidebar.warning(f"Could not load Clubbed Ensemble: {e}")

if saved_dir.exists():
    # 2. Load the regional PyTorch model explicitly.
    pt_file = saved_dir / "multitask_model.pt"
    if pt_file.exists():
        display_name = "PyTorch Regional Model"
        try:
            m, maps, sc = load_pytorch_model(pt_file)
            loaded_models_registry[display_name] = {
                "type": "pytorch",
                "model": m,
                "mappings": maps,
                "scaler": sc,
                "file": pt_file.name
            }
        except Exception as e:
            st.sidebar.warning(f"Could not load {pt_file.name}: {e}")

    # 3. Load the regional Keras model tied to metadata0 explicitly.
    keras_file = saved_dir / "multitask_model0.keras"
    metadata_file = saved_dir / "multitask_model_metadata0.json"
    if keras_file.exists() and metadata_file.exists():
        display_name = "TensorFlow Regional Model (metadata0)"
        try:
            m, maps, sc = load_keras_model(keras_file, metadata_file)
            loaded_models_registry[display_name] = {
                "type": "keras",
                "model": m,
                "mappings": maps,
                "scaler": sc,
                "file": keras_file.name
            }
        except Exception as e:
            st.sidebar.warning(f"Could not load {keras_file.name}: {e}")

# Sidebar Options
st.sidebar.header("⚙️ System & Farmer Profile Settings")

if not loaded_models_registry:
    st.error("⚠️ No trained models found!")
    st.stop()

available_model_names = list(loaded_models_registry.keys())
selected_model_name = st.sidebar.selectbox("🧠 Select Model Checkpoint", available_model_names)

active_model_data = loaded_models_registry[selected_model_name]
active_mappings = active_model_data["mappings"]
active_scaler = active_model_data["scaler"]
active_model_type = active_model_data["type"]
active_model_obj = active_model_data["model"]

districts = active_mappings["districts"]
soils = active_mappings["soils"]
crops = active_mappings["crops"]
fertilizers = active_mappings["fertilizers"]
explain_svc, cf_svc = load_xai_services(tuple(crops))

# --- Ensemble mapping-health indicator (sidebar) ---
# Surfaces the "silent regional-only fallback" failure mode instead of hiding it.
if active_model_type == "clubbed_ensemble":
    if getattr(active_model_obj, "mapping_loaded", False):
        st.sidebar.success("✅ Cross-dataset crop mapping loaded — dual-stream fusion active.")
    else:
        st.sidebar.error(
            "⚠️ No regional↔agronomic crop mapping found "
            f"(source: {getattr(active_model_obj, 'mapping_source', 'unknown')}). "
            "Every crop will fall back to regional-only scoring. Check that "
            "`ensemble_metadata.json` exists in `saved_models_clubbed/` with a populated "
            "`mappings.regional_to_agronomic` dictionary."
        )

# 5. Smarter defaults: Farmer Goal & Crop Type Filter
st.sidebar.subheader("🎯 Farmer Goals & Category Filters")
farmer_goal = st.sidebar.selectbox(
    "🏆 Recommendation Priority Goal",
    ["Balanced Growth (ML + Regional Context)", "Low Risk / High Stability", "Maximize Profit Potential"]
)

crop_type_options = ["All"] + sorted(list(set(CROP_TYPES_MAP.values())))
selected_crop_type = st.sidebar.selectbox("🏷️ Filter by Crop Category", crop_type_options)

st.sidebar.markdown("---")
st.sidebar.success(f"Active Inference Model: **{selected_model_name}**")

# ==========================================
# Session state initialization (single source of truth per field)
# ==========================================
st.session_state.setdefault("temp_val", 25.0)
st.session_state.setdefault("rainfall_val", 1000.0)
st.session_state.setdefault("rainfall_source", "manual")
st.session_state.setdefault("humidity_val", 65.0)

# ==========================================
# 4. Input Fields
# ==========================================
st.subheader("📊 Enter Location & Soil Parameters")

col1, col2 = st.columns(2)

with col1:
    district_val = st.selectbox("📍 Select District Name", districts)
    soil_val = st.selectbox("🪨 Select Soil Color / Type", soils)
    N = st.number_input("🧪 Nitrogen (N) content in soil (kg/ha)", min_value=0.0, value=50.0, step=1.0)
    P = st.number_input("🧪 Phosphorus (P) content in soil (kg/ha)", min_value=0.0, value=50.0, step=1.0)

# Fetch weather for the selected district (used for humidity/temp defaults and the widget below)
weather_data = weather_service.fetch_live_weather(district_val)

with col2:
    K = st.number_input("🧪 Potassium (K) content in soil (kg/ha)", min_value=0.0, value=50.0, step=1.0)
    ph = st.number_input("🌡️ Soil pH level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

    # --- Rainfall input: widget key IS the value we update, so the callback
    #     below and this widget can never disagree with each other. ---
    rainfall_label = "🌧️ Annual Rainfall (mm/year)"
    if st.session_state["rainfall_source"] == "auto":
        rainfall_label += " ✅ Auto-filled from historical data"

    rainfall = st.number_input(
        rainfall_label,
        min_value=0.0,
        step=10.0,
        key="rainfall_val",
        help="Enter the total annual rainfall (mm/year) for this region. The 7-day weather "
             "widget below shows a SHORT-TERM weekly forecast — that is NOT the annual value. "
             "Use the 'Calculate' button inside the weather widget to auto-fill the correct "
             "annual figure from the last ~3 months of historical data (extrapolated to a year)."
    )

    temp = st.number_input(
        "🌡️ Average Temperature (°C)", min_value=-10.0, max_value=60.0,
        value=float(st.session_state.get("temp_val", 25.0)), step=0.5
    )

    # Real humidity input (previously hardcoded to 70.0 and silently fed into the
    # agronomic model, biasing 28% of every fused prediction). Defaults from live
    # weather when available, editable by the farmer otherwise.
    humidity_default = float(weather_data.get("humidity", 65.0)) if weather_data.get("success") else st.session_state["humidity_val"]
    humidity = st.number_input(
        "💧 Relative Humidity (%)", min_value=0.0, max_value=100.0,
        value=humidity_default, step=1.0,
        help="Defaults to the current live reading for the selected district; edit if you have a more accurate local value."
    )
    st.session_state["humidity_val"] = humidity

# If a manual edit changed the value away from what an "auto" fill set, drop back to "manual" labeling
if st.session_state["rainfall_source"] == "auto" and "_rainfall_auto_value" in st.session_state:
    if abs(st.session_state["rainfall_val"] - st.session_state["_rainfall_auto_value"]) > 1e-6:
        st.session_state["rainfall_source"] = "manual"

# ==========================================
# Callback for the "Calculate Annual Rainfall" button.
# Runs BEFORE the rerun that Streamlit performs after any widget interaction,
# so session_state["rainfall_val"] is guaranteed correct the moment the
# number_input above re-renders -- no separate mirrored key, no manual st.rerun().
# ==========================================
def _apply_annual_rainfall(district: str):
    with st.spinner(f"Fetching ~3 months of historical precipitation for {district}..."):
        annual_data = weather_service.fetch_annual_rainfall(district)

    if annual_data.get("success"):
        st.session_state["rainfall_val"] = float(annual_data["annual_rainfall_mm"])
        st.session_state["_rainfall_auto_value"] = float(annual_data["annual_rainfall_mm"])
        st.session_state["rainfall_source"] = "auto"
        st.session_state["_annual_rainfall_result"] = annual_data
        st.session_state.pop("_annual_rainfall_error", None)
    else:
        st.session_state["_annual_rainfall_error"] = annual_data.get("error", "Unknown error")
        st.session_state.pop("_annual_rainfall_result", None)


def _sync_live_temp(current_temp: float):
    st.session_state["temp_val"] = float(current_temp)
    st.session_state["_temp_synced"] = current_temp


# Open-Meteo Weather Expander
if weather_data.get("success"):
    with st.expander(f"🌤️ Live Open-Meteo Weather & Agrometeorology — {district_val} District", expanded=True):
        w_col1, w_col2, w_col3, w_col4 = st.columns(4)
        w_col1.metric("🌡️ Live Temp", f"{weather_data['current_temp']} °C", f"Feels: {weather_data['feels_like']} °C")
        w_col2.metric("💧 Humidity", f"{weather_data['humidity']} %", f"Soil Moisture: {weather_data['soil_moisture_0_1cm']:.2f}")
        w_col3.metric("🌧️ 7-Day Rain", f"{weather_data['precip_7d_total']} mm", f"ET0: {weather_data['et0_7d_total']} mm")
        w_col4.metric("🌤️ Condition", weather_data["condition_title"], f"Wind: {weather_data['wind_speed']} km/h")

        # Critical unit mismatch warning
        st.warning(
            "⚠️ **Note:** The **7-Day Rain** shown above is the **short-term weekly forecast** "
            f"({weather_data['precip_7d_total']} mm over next 7 days) — "
            "**not** the annual rainfall figure needed by the model. "
            "The model requires **Annual Rainfall (mm/year)** which is typically **300–1700 mm** for Maharashtra districts. "
            "Please enter it manually in the input field above, or click **'📊 Calculate Annual Rainfall'** below to auto-fill it from historical data."
        )

        if weather_data.get("agri_alerts"):
            for alert in weather_data["agri_alerts"]:
                st.warning(alert)
        else:
            st.info(f"✅ Weather conditions in **{district_val}** are currently favorable for regular field operations.")

        btn_c1, btn_c2 = st.columns([1, 1])
        with btn_c1:
            st.button(
                "⚡ Sync Live Open-Meteo Temp to Input",
                use_container_width=True,
                on_click=_sync_live_temp,
                args=(weather_data["current_temp"],),
            )
            if st.session_state.get("_temp_synced") is not None:
                st.success(f"Synced Temperature to {st.session_state['_temp_synced']} °C!")

        with btn_c2:
            st.button(
                "📊 Calculate Annual Rainfall (Last 12 Months)",
                use_container_width=True,
                on_click=_apply_annual_rainfall,
                args=(district_val,),
            )
            if st.session_state.get("_annual_rainfall_result"):
                r = st.session_state["_annual_rainfall_result"]
                st.success(
                    f"✅ Auto-filled Annual Rainfall: **{r['annual_rainfall_mm']} mm/year**  \n"
                    f"📐 Method: {r.get('method', 'Extrapolation')}  \n"
                    f"📅 Based on last **{r.get('months_used', '~3')} months** ({r.get('days_used', '?')} days) of data  \n"
                    f"🌧️ Actual rain in that period: {r.get('total_past_mm', '?')} mm"
                )
            if st.session_state.get("_annual_rainfall_error"):
                st.error(f"❌ Could not fetch historical data: {st.session_state['_annual_rainfall_error']}")

        with st.expander("📅 View 7-Day Detailed Open-Meteo Forecast"):
            st.dataframe(pd.DataFrame(weather_data["forecast_7d"]), use_container_width=True)

# 5. UI Smart Defaults Preview: Top Local Crops
with st.expander(f"📍 View Historically Dominant Crops in {district_val}"):
    top_local = reg_service.get_top_district_crops(district_val, top_n=5)
    top_df = pd.DataFrame(top_local)[["crop", "crop_type", "area_share_pct", "avg_yield_ha"]]
    top_df.columns = ["Crop Name", "Category", "District Area Share (%)", "Avg Yield (t/ha)"]
    st.dataframe(top_df, use_container_width=True)

# ==========================================
# 5. Recommendation Button & Prediction Logic
# ==========================================
st.markdown("---")
if st.button("🚀 Generate Data-Backed Recommendation", type="primary", use_container_width=True):
    with st.spinner("Analyzing soil agronomics and querying 1997-2023 regional production data..."):
        dist_idx = districts.index(district_val)
        soil_idx = soils.index(soil_val)

        mean = np.array(active_scaler["mean"])
        scale = np.array(active_scaler["scale"])
        features = [N, P, K, ph, rainfall, temp]
        scaled_features = (np.array(features) - mean) / scale

        # fusion_map: crop -> "dual_stream" | "regional_only" | "single_model"
        fusion_map = {}
        mapping_loaded = True

        if active_model_type == "clubbed_ensemble":
            res_dict = active_model_obj.predict(
                district=district_val,
                soil=soil_val,
                nitrogen=N,
                phosphorus=P,
                potassium=K,
                ph=ph,
                rainfall=rainfall,
                temperature=temp,
                humidity=humidity,
                top_k=len(crops)
            )
            crop_prob_map = {item["crop"]: item["confidence"] / 100.0 for item in res_dict["top_crops"]}
            fusion_map = {item["crop"]: item["fusion_status"] for item in res_dict["top_crops"]}
            mapping_loaded = res_dict.get("mapping_loaded", True)
            crop_probs_t = np.array([crop_prob_map.get(c, 0.0) for c in crops])

            # The clubbed ensemble can package either a PyTorch or Keras
            # regional model.  Keras models require the three named inputs,
            # whereas the PyTorch model accepts positional tensors.
            if active_model_obj.is_keras_regional:
                regional_predictions = active_model_obj.reg_model({
                    "numerical": np.array([scaled_features], dtype=np.float32),
                    "district": np.array([dist_idx], dtype=np.int32),
                    "soil": np.array([soil_idx], dtype=np.int32),
                }, training=False)
                fert_probs_t = regional_predictions[1].numpy()[0]
            else:
                x_num = torch.FloatTensor([scaled_features])
                x_dist = torch.LongTensor([dist_idx])
                x_soil = torch.LongTensor([soil_idx])
                with torch.no_grad():
                    _, fert_logits = active_model_obj.reg_model(x_num, x_dist, x_soil)
                    fert_probs_t = torch.softmax(fert_logits, dim=1).numpy()[0]
        elif active_model_type == "pytorch":
            x_num = torch.FloatTensor([scaled_features])
            x_dist = torch.LongTensor([dist_idx])
            x_soil = torch.LongTensor([soil_idx])

            with torch.no_grad():
                crop_logits, fert_logits = active_model_obj(x_num, x_dist, x_soil)
                crop_probs_t = torch.softmax(crop_logits, dim=1).numpy()[0]
                fert_probs_t = torch.softmax(fert_logits, dim=1).numpy()[0]
            fusion_map = {c: "single_model" for c in crops}
        else:
            inputs = {
                "numerical": np.array([scaled_features], dtype=np.float32),
                "district": np.array([[dist_idx]], dtype=np.int32),
                "soil": np.array([[soil_idx]], dtype=np.int32)
            }
            predictions = active_model_obj(inputs, training=False)
            crop_probs_t, fert_probs_t = predictions[0].numpy()[0], predictions[1].numpy()[0]
            fusion_map = {c: "single_model" for c in crops}

        # Process all crop probabilities with regional statistics
        crop_candidates = []
        for c_idx, prob in enumerate(crop_probs_t):
            crop_name = crops[c_idx]

            # Apply crop category filter if selected
            if selected_crop_type != "All" and CROP_TYPES_MAP.get(crop_name) != selected_crop_type:
                continue

            yield_info = reg_service.get_expected_yield(crop_name, district_val)
            risk_info = reg_service.get_yield_volatility_risk(crop_name, district_val)
            plausibility = reg_service.check_regional_plausibility(crop_name, district_val)
            profit_info = reg_service.get_profit_estimate(crop_name, district_val)

            # Goal-based score adjustment
            ml_score = float(prob) * 100.0
            if farmer_goal == "Low Risk / High Stability":
                # Penalize high volatility CV score
                adjusted_score = ml_score * (1.0 / (1.0 + risk_info["cv_score"]))
            elif farmer_goal == "Maximize Profit Potential":
                # Boost higher net profit per acre
                profit_factor = max(0.1, profit_info["net_profit_acre"] / 30000.0)
                adjusted_score = ml_score * (0.6 + 0.4 * min(2.0, profit_factor))
            else:
                adjusted_score = ml_score

            fusion_status = fusion_map.get(crop_name, "unknown")
            if fusion_status == "dual_stream":
                fusion_badge = "🟢 Dual-Stream"
            elif fusion_status == "regional_only":
                fusion_badge = "🟡 Regional-Only Fallback"
            elif fusion_status == "single_model":
                fusion_badge = "⚪ Single-Model"
            else:
                fusion_badge = "❔ Unknown"

            crop_candidates.append({
                "crop": crop_name,
                "ml_confidence": ml_score,
                "adjusted_score": adjusted_score,
                "yield_t_ha": yield_info["yield_tonne_per_ha"],
                "yield_q_acre": yield_info["yield_quintal_per_acre"],
                "yield_source": yield_info["data_source"],
                "cv_score": risk_info["cv_score"],
                "risk_badge": risk_info["risk_badge"],
                "risk_tier": risk_info["risk_tier"],
                "area_share_pct": plausibility["area_share_pct"],
                "is_plausible": plausibility["is_plausible"],
                "plausibility_badge": plausibility["plausibility_badge"],
                "warning_message": plausibility["warning_message"],
                "net_profit_acre": profit_info["net_profit_acre"],
                "category": CROP_TYPES_MAP.get(crop_name, "Other"),
                "fusion_status": fusion_status,
                "fusion_badge": fusion_badge,
            })

        crop_candidates.sort(key=lambda x: x["adjusted_score"], reverse=True)

        if not crop_candidates:
            st.warning(f"No crops match the selected category filter: '{selected_crop_type}'. Showing all crops instead.")
            st.stop()

        top_rec = crop_candidates[0]
        top_fert_idx = np.argmax(fert_probs_t)
        top_fert = fertilizers[top_fert_idx]
        top_fert_conf = fert_probs_t[top_fert_idx] * 100.0
        # Persist the actual model result so later chat messages remain grounded
        # even after Streamlit reruns following a chat submission.
        st.session_state["last_recommendation"] = {
            **top_rec,
            "fertilizer": top_fert,
        }

        # Explain the underlying model's crop probabilities, rather than the
        # downstream business/risk-adjusted ranking shown in the comparison.
        predict_fn = make_predict_fn(
            active_model_type, active_model_obj, dist_idx, soil_idx,
            mean, scale, crops,
        )
        current_x = np.array([N, P, K, ph, rainfall, temp, humidity], dtype=float)

        # ==========================================
        # 6. Display Recommendations & Analytical Results
        # ==========================================
        st.success("🎉 Data-Backed Recommendation Generated Successfully!")

        if active_model_type == "clubbed_ensemble" and not mapping_loaded:
            st.error(
                "⚠️ **Ensemble degraded:** no regional↔agronomic crop mapping was loaded, so "
                "every crop below is Regional-Only rather than true dual-stream fusion. "
                "See the sidebar for details."
            )

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown(
                f"""
                <div style="background-color:rgba(46, 204, 113, 0.1); padding: 25px; border-radius: 12px; border-left: 6px solid #2ecc71;">
                  <h3 style="color:#2ecc71; margin-top:0;">🌱 Top Recommended Crop</h3>
                  <h1 style="margin: 10px 0;">{top_rec['crop'].upper()}</h1>
                  <p style="font-size: 16px; margin: 0;">Soil Model Match: <strong>{top_rec['ml_confidence']:.2f}%</strong></p>
                  <p style="font-size: 14px; margin-top: 5px; color:#555;">Category: <strong>{top_rec['category']}</strong></p>
                  <p style="font-size: 14px; margin-top: 5px; color:#555;">Ensemble Fusion: <strong>{top_rec['fusion_badge']}</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with res_col2:
            st.markdown(
                f"""
                <div style="background-color:rgba(52, 152, 219, 0.1); padding: 25px; border-radius: 12px; border-left: 6px solid #3498db;">
                  <h3 style="color:#3498db; margin-top:0;">🧪 Recommended Fertilizer</h3>
                  <h1 style="margin: 10px 0;">{top_fert.upper()}</h1>
                  <p style="font-size: 16px; margin: 0;">Fertilizer Model Match: <strong>{top_fert_conf:.2f}%</strong></p>
                  <p style="font-size: 14px; margin-top: 5px; color:#555;">Target Soil: <strong>{soil_val}</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Regional Plausibility & Risk Alerts
        st.markdown("### 🔍 Regional Validation & Risk Assessment")

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("🌾 Expected Yield", f"{top_rec['yield_t_ha']} t/ha", f"{top_rec['yield_q_acre']} Quintals/Acre")
        m_col2.metric("📈 Yield Volatility (CV)", f"{top_rec['cv_score']:.2f}", top_rec['risk_badge'])
        m_col3.metric("📊 Regional Area Share", f"{top_rec['area_share_pct']:.1f}%", f"{district_val} District")
        m_col4.metric("💰 Est. Net Profit", f"₹{top_rec['net_profit_acre']:,.0f} / Acre", "Historical Trend")

        if active_model_type == "clubbed_ensemble":
            if top_rec["fusion_status"] == "regional_only":
                st.info(
                    "🟡 **Fusion note:** this crop had no matching entry in the agronomic (22-crop) "
                    "dataset, so its score comes from the regional district model alone — the "
                    "agronomic model was not consulted for this specific crop."
                )
            elif top_rec["fusion_status"] == "dual_stream":
                st.info(
                    f"🟢 **Fusion note:** this score blends both models "
                    f"({active_model_obj.district_weight*100:.0f}% regional + "
                    f"{active_model_obj.agronomic_weight*100:.0f}% agronomic)."
                )

        # 3. Plausibility Flag Caveat
        if not top_rec["is_plausible"] and top_rec["warning_message"]:
            st.warning(f"⚠️ **Regional Plausibility Caveat**: {top_rec['warning_message']}")
        else:
            st.info(f"✅ **Regional Verification**: {top_rec['plausibility_badge']}. This crop is historically established in {district_val}.")

        # 6. Top-3 Alternative Crops Comparison Table
        st.markdown("### 📑 Top Alternative Crops Comparison")
        alt_data = []
        for item in crop_candidates[:4]:
            alt_data.append({
                "Crop": item["crop"],
                "Category": item["category"],
                "ML Match": f"{item['ml_confidence']:.1f}%",
                "Expected Yield": f"{item['yield_t_ha']} t/ha ({item['yield_q_acre']} q/acre)",
                "Risk Rating": item["risk_badge"],
                "Regional Area Share": f"{item['area_share_pct']:.1f}%",
                "Regional Plausibility": item["plausibility_badge"],
                "Est. Profit / Acre": f"₹{item['net_profit_acre']:,.0f}",
                "Fusion Type": item["fusion_badge"],
            })

        st.dataframe(pd.DataFrame(alt_data), use_container_width=True)

        with st.expander("🧠 Why this crop? (SHAP explanation)", expanded=True):
            try:
                explanation = explain_svc.explain(
                    predict_fn, current_x, target_crop=top_rec["crop"], nsamples=24
                )
                st.write(explanation["natural_language_summary"])
                st.dataframe(pd.DataFrame(explanation["contributions"]), use_container_width=True)
            except Exception as exc:
                st.warning(f"The SHAP explanation could not be generated: {exc}")

        st.markdown("### 🔄 Why not the alternatives? (Counterfactual analysis)")
        for alternative in crop_candidates[1:4]:
            with st.expander(f"Why not {alternative['crop'].title()}?"):
                try:
                    rejection = explain_svc.explain_rejection(
                        predict_fn, current_x, top_rec["crop"], alternative["crop"], nsamples=24
                    )
                    st.write(rejection["natural_language_summary"])

                    counterfactual = cf_svc.find_counterfactual(
                        predict_fn, current_x, target_crop=alternative["crop"]
                    )
                    if counterfactual["achieved"]:
                        st.success(
                            f"**{counterfactual['feasibility_label']}** — target crop "
                            f"probability would rise from "
                            f"{counterfactual['starting_target_probability_pct']}% to "
                            f"{counterfactual['final_target_probability_pct']}%"
                        )
                        st.dataframe(pd.DataFrame(counterfactual["deltas"]), use_container_width=True)
                    else:
                        st.warning(
                            f"{counterfactual['feasibility_label']} — final target probability "
                            f"would be {counterfactual['final_target_probability_pct']}%. "
                            "No realistic change to N/P/K/pH/rainfall alone gets there."
                        )
                except Exception as exc:
                    st.warning(f"Alternative analysis could not be generated: {exc}")

        if active_model_type == "clubbed_ensemble":
            st.caption(
                "🟢 Dual-Stream = both regional and agronomic models contributed to this score.  "
                "🟡 Regional-Only Fallback = no agronomic-dataset counterpart was found for this crop; "
                "the regional model had full authority.  "
                "Check the sidebar mapping-health indicator if you see mostly 🟡 badges."
            )

# The sidebar keeps chat available without taking space from the analytical
# recommendation view. Chat reruns do not re-run SHAP/counterfactual analysis.
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

app_context = None
if "last_recommendation" in st.session_state:
    rec = st.session_state["last_recommendation"]
    app_context = {
        "District": district_val,
        "Recommended Crop": f"{rec['crop']} ({rec['ml_confidence']:.1f}% match)",
        "Recommended Fertilizer": rec.get("fertilizer", "N/A"),
        "Regional plausibility": rec.get("plausibility_badge", "N/A"),
    }

with st.sidebar:
    st.markdown("---")
    st.subheader("💬 KrushiDaata Sahayak")
    st.caption("Ask about crops, fertilizers, schemes, or your latest recommendation.")

    for msg in st.session_state["chat_messages"][-8:]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_q = st.chat_input("Ask KrushiDaata Sahayak...", key="sidebar_chat_input")
    if user_q:
        st.session_state["chat_messages"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.write(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = chatbot.chat(
                    user_q,
                    chat_history=st.session_state["chat_messages"][:-1],
                    app_context=app_context,
                )
            if result["success"]:
                st.write(result["reply"])
                st.session_state["chat_messages"].append(
                    {"role": "assistant", "content": result["reply"]}
                )
            else:
                st.error(result["error"])

# Footer
st.markdown("---")
st.caption(f"KrushiDaata Multi-Task Deep Learning & Regional Production Analytics - Active Engine: {selected_model_name}")
