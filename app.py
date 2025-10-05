"""FastAPI application exposing the exoplanet classifier as a service."""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from exoplanet_app import preprocess, train

app = FastAPI(title="Exoplanet Classifier", version="1.0.0")


def _load_model_payload() -> Dict[str, Any]:
    try:
        return train.get_trained_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model not trained yet.") from exc


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    model_payload = _load_model_payload()

    try:
        dataframe = pd.read_csv(io.BytesIO(content))
    except Exception as exc:  # pragma: no cover - FastAPI will turn into 400
        raise HTTPException(status_code=400, detail="Unable to parse CSV file.") from exc

    features = preprocess.prepare_features_frame(dataframe)
    predictions = model_payload["model"].predict(features)

    records: List[Dict[str, Any]] = []
    source_records = dataframe.to_dict(orient="records")
    for row, prediction in zip(source_records, predictions):
        enriched = dict(row)
        enriched["prediction"] = prediction
        records.append(enriched)

    return {"results": records, "count": len(records)}


@app.post("/retrain")
async def retrain(
    file: UploadFile = File(...),
    algorithm: str = "random_forest",
    base_model: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    base_model_bytes = await base_model.read() if base_model is not None else None

    try:
        metrics = train.retrain_model(content, algorithm=algorithm, base_model=base_model_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - bubble unexpected errors
        raise HTTPException(status_code=500, detail="Retraining failed.") from exc

    return {"status": "retrained", "metrics": metrics}


@app.get("/stats")
async def stats() -> Dict[str, Any]:
    try:
        metrics = train.get_stored_metrics()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No metrics available. Train the model first.") from exc
    return metrics
