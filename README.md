# Exoplanet Classifier

A small FastAPI + Streamlit project for training and serving an exoplanet disposition classifier that merges the TESS, Kepler, and K2 catalogues. Run the commands below from the project root (the folder that contains the `exoplanet_app/` package).

## Project Structure
- `data/` – bundled CSV catalogues used for the baseline training run.
- `models/` – persisted `joblib` payload (`exoplanet_model.pkl`) and metrics generated after training.
- `preprocess.py` – helpers for loading and cleaning the catalogues.
- `train.py` – model training, retraining, and persistence utilities.
- `app.py` – FastAPI service exposing `/predict`, `/retrain`, and `/stats`.
- `ui.py` – Streamlit dashboard for uploads, training, predictions, and insights.

## Prerequisites
- Python 3.10+
- Recommended: create a virtual environment before installing dependencies.

```bash
cd /path/to/project/root
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Train the Baseline Model
Run the bundled training entry point to fit a model on the packaged datasets and store the artefacts under `models/`:

```bash
python -m exoplanet_app.train
```

By default the script uses the Random Forest baseline. To switch algorithms or reuse a saved checkpoint, update the Streamlit UI or call the training helpers from Python (see below). The command prints accuracy, recall, and F1 metrics to the console and writes the model/metrics files for later use.

## Run the FastAPI Service
Serve the prediction API locally (default http://127.0.0.1:8000):

```bash
uvicorn exoplanet_app.app:app --reload
```

Endpoints:
- `POST /predict` – upload a CSV to receive predicted dispositions.
- `POST /retrain` – upload additional labelled data to extend the baseline model.
- `GET /stats` – retrieve the most recent training metrics.

## Launch the Streamlit UI
In a second terminal (with the same virtual environment activated), start the dashboard:

```bash
streamlit run exoplanet_app/ui.py
```

The UI includes three tabs:
- **Training** – trigger baseline training or upload a CSV to retrain.
- **Prediction** – upload new candidate records and view predictions in a table.
- **Insights** – inspect dataset summaries, an orbital feature scatter, and a star map built from RA/Dec positions in the training catalogues, all rendered in a cosmic-themed dark UI.

### Choose Algorithms & Base Models
- **Training tab**: pick between the Random Forest baseline and an XGBoost ensemble.
- Optionally upload a previously saved model (`.pkl` joblib payload). The app reuses its hyperparameters; for XGBoost, the booster is also warm-started with your new data.
- API users can supply `algorithm` (\`random_forest\` or \`xgboost\`) and an optional `base_model` file when calling `POST /retrain`.

## Updating the Model with New Data
1. Obtain a CSV with the required feature columns (`pl_orbper`, `pl_orbsmax`, ... , `st_logg`) and any of the supported disposition labels (`tfopwg_disp`, `disposition`, `koi_disposition`, or `koi_pdisposition`).
2. Use either the Streamlit Training tab or the FastAPI `/retrain` endpoint to upload the file. The service will merge the data with the bundled catalogues, retrain the classifier, and refresh the saved model/metrics.
3. Confirm the updated metrics through the UI or via `GET /stats`.

## Development Notes
- If you update the dependencies, regenerate the virtual environment and reinstall via `pip install -r requirements.txt`.
- Clean model artefacts by removing files from `models/` before running a fresh baseline training session, if desired.
