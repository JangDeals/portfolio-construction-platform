"""
Model and object serialization module.
Saves trained models (and their metadata) to disk using Joblib, and
provides matching load functions, so models can be reused without
retraining.
"""

import joblib
import json
from datetime import datetime
from pathlib import Path
import sys
sys.path.append("..")
from src.utils.paths import MODELS_DIR


def save_model(model, model_name: str, feature_cols: list[str], metadata: dict = None) -> None:
    """
    Save a trained model to models/, along with a metadata JSON recording
    its feature configuration and any additional context.

    Parameters
    ----------
    model : fitted scikit-learn estimator
    model_name : str
        Used as the base filename (no extension).
    feature_cols : list[str]
        Exact feature columns and order the model expects at prediction time.
    metadata : dict, optional
        Additional context: training date range, validated performance,
        asset name, etc.

    Returns
    -------
    None
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{model_name}.pkl"
    joblib.dump(model, model_path)

    full_metadata = {
        "model_name": model_name,
        "saved_timestamp": datetime.now().isoformat(),
        "feature_cols": feature_cols,
        "model_type": type(model).__name__,
    }
    if metadata:
        full_metadata.update(metadata)

    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(full_metadata, f, indent=4)


def load_model(model_name: str) -> tuple:
    """
    Load a previously saved model and its metadata.

    Parameters
    ----------
    model_name : str

    Returns
    -------
    tuple
        (model, metadata dict)

    Raises
    ------
    FileNotFoundError
        If the model or its metadata file does not exist.
    """
    model_path = MODELS_DIR / f"{model_name}.pkl"
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Model or metadata not found for: {model_name}")

    model = joblib.load(model_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return model, metadata



def save_walk_forward_models(simple_returns, initial_train_years: int = 5, target_horizon: int = 21) -> dict:
    """
    Retrain and save EVERY walk-forward iteration's Linear Regression
    volatility model, for every asset, so each historical rebalancing
    decision's exact model can be audited later without rerunning the
    full training loop.

    Parameters
    ----------
    simple_returns : pd.DataFrame
    initial_train_years : int, default 5
    target_horizon : int, default 21

    Returns
    -------
    dict
        Summary of all models saved: {model_name: metadata}.
    """
    from src.forecasting.volatility_forecasting import build_volatility_features
    from sklearn.linear_model import LinearRegression
    import pandas as pd

    feature_cols = ["trailing_vol_21d", "trailing_vol_60d", "trailing_mean_abs_return_21d", "trailing_return_21d"]
    saved_summary = {}

    for ticker in simple_returns.columns:
        feature_df = build_volatility_features(simple_returns[ticker], target_horizon)
        start_date = feature_df.index.min()
        first_test_date = start_date + pd.DateOffset(years=initial_train_years)
        test_years = pd.date_range(start=first_test_date, end=feature_df.index.max(), freq="YS")

        for year_start in test_years:
            train_data = feature_df[feature_df.index < year_start]
            if len(train_data) < 100:
                continue

            model = LinearRegression()
            model.fit(train_data[feature_cols], train_data["target_volatility"])

            model_name = f"volatility_{ticker}_{year_start.date()}"
            metadata = {
                "ticker": ticker,
                "train_end_date": str(year_start.date()),
                "train_rows": len(train_data),
                "target_horizon_days": target_horizon,
            }
            save_model(model, model_name, feature_cols, metadata)
            saved_summary[model_name] = metadata

    return saved_summary