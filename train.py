"""Model training utilities for the exoplanet classifier."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, IO, Optional, Tuple, Union

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

try:  # Optional dependency handled at runtime.
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover - handled in runtime checks and missing libomp
    xgb = None

import preprocess

ModelSource = Union[str, Path, bytes, IO[bytes], None]

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_FILENAME = "exoplanet_model.pkl"
METRICS_FILENAME = "metrics.json"


def _ensure_models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def _default_model_path() -> Path:
    return _ensure_models_dir() / MODEL_FILENAME


def _metrics_path() -> Path:
    return _ensure_models_dir() / METRICS_FILENAME


def _save_model(model: Any, algorithm: str, model_path: Optional[Path] = None) -> Path:
    path = model_path or _default_model_path()
    payload = {
        "model": model,
        "features": preprocess.FEATURE_COLUMNS,
        "algorithm": algorithm,
    }
    joblib.dump(payload, path)
    return path


def _save_metrics(metrics: Dict[str, float]) -> Path:
    path = _metrics_path()
    path.write_text(json.dumps(metrics, indent=2))
    return path


def _read_additional_csv(source: Union[str, Path, bytes, IO[bytes]]) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        return pd.read_csv(source, comment="#", skipinitialspace=True)

    if isinstance(source, bytes):
        return pd.read_csv(io.BytesIO(source), comment="#", skipinitialspace=True)

    if hasattr(source, "read"):
        try:
            source.seek(0)
        except (AttributeError, io.UnsupportedOperation):
            pass
        return pd.read_csv(source, comment="#", skipinitialspace=True)

    raise TypeError("Unsupported type for new_csv. Provide a path, bytes, or a file-like object.")


def _infer_algorithm_from_model(model: Any) -> str:
    if isinstance(model, RandomForestClassifier):
        return "random_forest"
    if xgb is not None and isinstance(model, xgb.XGBClassifier):  # type: ignore[attr-defined]
        return "xgboost"
    return "unknown"


def _normalise_base_payload(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        algorithm = obj.get("algorithm") or _infer_algorithm_from_model(model)
    else:
        model = obj
        algorithm = _infer_algorithm_from_model(model)

    if algorithm not in {"random_forest", "xgboost"}:
        raise ValueError("Unsupported base model type. Expected RandomForestClassifier or XGBClassifier.")

    return {"model": model, "algorithm": algorithm}


def _load_base_model(source: ModelSource) -> Optional[Dict[str, Any]]:
    if source is None:
        return None

    if isinstance(source, (str, Path)):
        obj = joblib.load(source)
        return _normalise_base_payload(obj)

    if isinstance(source, bytes):
        buffer = io.BytesIO(source)
        obj = joblib.load(buffer)
        return _normalise_base_payload(obj)

    if hasattr(source, "read"):
        data = source.read()
        return _load_base_model(data)

    raise TypeError("Unsupported base model source. Provide path, bytes, or file-like object.")


def _initialise_model(
    algorithm: str,
    random_state: int,
    base_payload: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    algorithm = algorithm.lower()

    if algorithm == "random_forest":
        params = {
            "n_estimators": 300,
            "random_state": random_state,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
        }
        if base_payload is not None:
            params.update(base_payload["model"].get_params())
            params.setdefault("n_jobs", -1)
            if params.get("random_state") is None:
                params["random_state"] = random_state
        model = RandomForestClassifier(**params)
        return model, {}

    if algorithm == "xgboost":
        if xgb is None:
            raise ImportError("xgboost is not installed. Install it to use the XGBoost model option.")

        params = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": random_state,
            "n_jobs": -1,
        }
        fit_kwargs: Dict[str, Any] = {}
        if base_payload is not None:
            params.update(base_payload["model"].get_params())
            fit_kwargs["xgb_model"] = base_payload["model"]
        model = xgb.XGBClassifier(**params)  # type: ignore[attr-defined]
        return model, fit_kwargs

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def train_model(
    df: pd.DataFrame,
    *,
    model_path: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    algorithm: str = "random_forest",
    base_model: ModelSource = None,
) -> Dict[str, float]:
    """Train a classifier on the provided DataFrame and persist it.

    Parameters
    ----------
    df:
        Prepared DataFrame containing the numeric features and label column.
    model_path:
        Optional override for where the trained model should be written.
    test_size:
        Hold-out fraction for validation metrics.
    random_state:
        RNG seed for reproducibility.
    algorithm:
        `"random_forest"` (default) or `"xgboost"`.
    base_model:
        Optional pre-trained model payload (path/bytes/file-like). When supplied,
        the model's hyperparameters are reused. For XGBoost, the booster is also
        used for warm-starting the new training run.
    """

    if df.empty:
        raise ValueError("Training DataFrame is empty.")

    base_payload = _load_base_model(base_model)

    if algorithm is None or algorithm == "auto":
        algorithm_name = base_payload["algorithm"] if base_payload else "random_forest"
    else:
        algorithm_name = algorithm.lower()

    if base_payload is not None and base_payload["algorithm"] != algorithm_name:
        raise ValueError(
            f"Base model type '{base_payload['algorithm']}' does not match requested algorithm '{algorithm_name}'."
        )

    X, y = preprocess.split_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    model, fit_kwargs = _initialise_model(algorithm_name, random_state, base_payload)
    model.fit(X_train, y_train, **fit_kwargs)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
    }

    _save_model(model, algorithm_name, model_path)
    _save_metrics(metrics)
    return metrics


def retrain_model(
    new_csv: Union[str, Path, bytes, IO[bytes]],
    *,
    algorithm: str = "random_forest",
    base_model: ModelSource = None,
) -> Dict[str, float]:
    """Load the base datasets, append *new_csv*, and retrain the model."""

    base_df = preprocess.load_datasets()
    new_df_raw = _read_additional_csv(new_csv)
    new_df = preprocess.prepare_dataframe(new_df_raw)

    combined_df = pd.concat([base_df, new_df], ignore_index=True, sort=False)
    metrics = train_model(combined_df, algorithm=algorithm, base_model=base_model)
    return metrics


def get_trained_model(model_path: Optional[Path] = None) -> Dict[str, Any]:
    """Return the persisted model payload."""

    path = model_path or _default_model_path()
    if not path.exists():
        raise FileNotFoundError("Model file not found. Train the model first.")
    return joblib.load(path)


def get_stored_metrics() -> Dict[str, float]:
    """Return the metrics recorded during the last training run."""

    path = _metrics_path()
    if not path.exists():
        raise FileNotFoundError("Metrics file not found. Train the model first.")
    return json.loads(path.read_text())


def train_from_base(
    algorithm: str = "random_forest",
    base_model: ModelSource = None,
) -> Dict[str, float]:
    """Train using the packaged datasets with optional algorithm/base model overrides."""

    base_df = preprocess.load_datasets()
    return train_model(base_df, algorithm=algorithm, base_model=base_model)


__all__ = [
    "train_model",
    "retrain_model",
    "get_trained_model",
    "get_stored_metrics",
    "train_from_base",
]


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    metrics = train_from_base()
    print("Training complete. Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")
