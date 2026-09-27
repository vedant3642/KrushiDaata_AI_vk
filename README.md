# KrushiDaata

KrushiDaata is a smart agriculture recommendation system that combines crop and fertilizer prediction with regional agronomic context, weather intelligence, and explainable AI insights for farmers in Maharashtra.

The project includes:
- Multi-task crop and fertilizer prediction models
- Ensemble model fusion for regional + agronomic reasoning
- Regional statistics and yield analytics
- Weather-based annual rainfall estimation
- SHAP-based explanations and counterfactual recommendations
- Chatbot assistance for farmer guidance
- Streamlit-based web application

## Overview

This project helps farmers make better agronomic decisions by predicting the most suitable crop and fertilizer based on soil, weather, and regional context. It blends:
- a regional model trained on district-specific agricultural patterns
- an agronomic model trained on crop suitability logic
- an ensemble layer that balances local expertise with biological crop compatibility

## Key Features

### Crop and fertilizer recommendation
- Predicts best crop based on N, P, K, pH, rainfall, temperature, and humidity
- Recommends fertilizer strategy using a multi-task neural model
- Supports multiple model backends such as PyTorch, Keras, and ensemble fusion

### Regional analytics
- Pulls historical yield, area, and production data by district and crop
- Provides plausibility checks against regional agricultural trends

### Weather integration
- Uses Open-Meteo weather data for seasonal and annual rainfall estimation
- Validates rainfall values against the expected annual rainfall range

### Explainability and decision support
- SHAP explanations for model predictions
- Counterfactual suggestions for exploring “what if” scenarios
- Farmer-friendly insights to understand model reasoning

### Chatbot assistance
- AI-style guidance layer powered by Groq
- Assists with crop-health and recommendation explanations

## Project Structure

```text
.
├── app.py                              # Streamlit web application
├── requirements.txt                   # Python dependencies
├── CLAUDE.md                          # Project guide and architecture notes
├── Crop and fertilizer dataset (1).csv
├── Crop_recommendation.csv
├── crop-wise-area-production-yield.csv
├── synthetic_crop_fertilizer_augmented.csv
├── evaluate_all_models_detailed.py     # Comprehensive model evaluation
├── compare_all_models.py               # Compare model outputs
├── evaluate_regional_sanity.py        # Regional validation checks
├── run_manual_pipeline.py             # Manual prediction pipeline
├── train_clubbed_ensemble.py          # Ensemble training
├── train_torch.py                     # PyTorch model training
├── train_tensor.py                    # TensorFlow/Keras model training
├── train_gan.py                       # GAN-related experiments
├── verify_saved_model.py              # Model verification checks
├── saved_models_2/                     # Regional model artifacts
├── saved_models_clubbed/              # Ensemble model artifacts
├── saved_gan_models/                  # GAN outputs
├── services/                          # Core backend services
│   ├── chatbot_service.py
│   ├── clubbed_prediction_service.py
│   ├── counterfactual_service.py
│   ├── explainability_service.py
│   ├── regional_stats_service.py
│   └── weather_service.py
└── .env                              # Local environment variables (if present)
```

## Model Architecture

The project uses a multi-task recommendation system designed around agronomic and regional signals.

### Regional model
The regional model is trained for Maharashtra-specific crop and fertilizer prediction using district and soil embeddings along with tabular agricultural features.

### Agronomic model
The agronomic stream uses crop suitability and climate behavior patterns to improve recommendations beyond local survey distributions.

### Ensemble model
The clubbed ensemble combines the strengths of both streams:
- strong regional context awareness
- biological agronomic validation
- improved robustness for out-of-distribution conditions

## Setup

### 1. Create virtual environment

```powershell
python -m venv venv
```

### 2. Activate environment

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root if needed. At minimum, the app may require:

```env
GROQ_API_KEY=your_api_key_here
```

## Run the Application

```powershell
venv\Scripts\python.exe -m streamlit run app.py
```

Or from the project root:

```powershell
.\venv\Scripts\streamlit run app.py
```

## Evaluation and Training

Run the benchmark scripts inside the project virtual environment:

```powershell
.\venv\Scripts\python.exe evaluate_all_models_detailed.py
.\venv\Scripts\python.exe compare_all_models.py
.\venv\Scripts\python.exe evaluate_regional_sanity.py
```

Train the models using the project scripts:

```powershell
.\venv\Scripts\python.exe train_torch.py
.\venv\Scripts\python.exe train_clubbed_ensemble.py
```

## Dependencies

Core dependencies include:
- Streamlit
- PyTorch
- TensorFlow
- NumPy
- Pandas
- scikit-learn
- SHAP
- Requests
- python-dotenv

## Notes

- The weather module uses the Open-Meteo archive API to estimate annual rainfall from the last 12 months.
- Annual rainfall is expected in mm/year, while the weather widget may display short-term precipitation in mm/day or mm over a 7-day window.
- The project is designed around agricultural decision support for farmers and agricultural planners in Maharashtra.

## License

This project does not currently declare a license file. Please check with the repository owner before commercial or public reuse.

## Contact and Usage

This project is intended for smart farming and agricultural research use cases, especially for crop recommendation and explainable AI in agronomy.
