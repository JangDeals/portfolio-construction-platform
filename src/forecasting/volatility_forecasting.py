"""
Volatility forecasting module.
Builds leakage-safe features and targets for forecasting next-period
realized volatility, and evaluates Linear Regression and Random Forest
models against a naive persistence baseline using walk-forward validation.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from src.portfolio.construction import inverse_volatility_portfolio


def build_volatility_features(returns: pd.Series, target_horizon: int = 21) -> pd.DataFrame:
    """
    Build leakage-safe features and a forward-looking target for
    volatility forecasting.

    Every feature is computed using only information available as of each
    date (trailing windows). The target is realized volatility over the
    NEXT target_horizon days, explicitly shifted forward so that, at each
    row, the target is genuinely unknown information at prediction time.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns for a single asset or portfolio.
    target_horizon : int, default 21
        Number of forward trading days over which the target volatility
        is realized (~1 month).

    Returns
    -------
    pd.DataFrame
        Feature columns plus a 'target_volatility' column. Rows with any
        NaN (from the initial trailing window or the final forward-looking
        window) are dropped.
    """
    df = pd.DataFrame(index=returns.index)

    # Features: all TRAILING (backward-looking) — safe, known as of "today"
    df["trailing_vol_21d"] = returns.rolling(21).std() * np.sqrt(252)
    df["trailing_vol_60d"] = returns.rolling(60).std() * np.sqrt(252)
    df["trailing_mean_abs_return_21d"] = returns.abs().rolling(21).mean()
    df["trailing_return_21d"] = (1 + returns).rolling(21).apply(np.prod, raw=True) - 1

    # Target: realized volatility over the NEXT target_horizon days —
    # deliberately shifted FORWARD, using .shift(-target_horizon), so this
    # value is genuinely unknown at the time each row's features are known.
    forward_vol = returns.rolling(target_horizon).std().shift(-target_horizon) * np.sqrt(252)
    df["target_volatility"] = forward_vol

    df = df.dropna()
    return df



def walk_forward_volatility_forecast(
    feature_df: pd.DataFrame,
    model_type: str = "linear",
    initial_train_years: int = 5,
) -> pd.DataFrame:
    """
    Evaluate a volatility forecasting model using walk-forward validation:
    train on an expanding window, predict the following year, and repeat.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Output of build_volatility_features.
    model_type : str, default "linear"
        "linear" or "random_forest".
    initial_train_years : int, default 5

    Returns
    -------
    pd.DataFrame
        Actual vs. predicted target_volatility for every out-of-sample date,
        plus the naive persistence baseline (trailing_vol_21d) for comparison.
    """
    feature_cols = ["trailing_vol_21d", "trailing_vol_60d", "trailing_mean_abs_return_21d", "trailing_return_21d"]

    start_date = feature_df.index.min()
    first_test_date = start_date + pd.DateOffset(years=initial_train_years)
    test_years = pd.date_range(start=first_test_date, end=feature_df.index.max(), freq="YS")

    all_predictions = []

    for i, year_start in enumerate(test_years):
        train_data = feature_df[feature_df.index < year_start]
        year_end = test_years[i + 1] if i + 1 < len(test_years) else feature_df.index.max() + pd.Timedelta(days=1)
        test_data = feature_df[(feature_df.index >= year_start) & (feature_df.index < year_end)]

        if len(test_data) == 0 or len(train_data) == 0:
            continue

        X_train, y_train = train_data[feature_cols], train_data["target_volatility"]
        X_test, y_test = test_data[feature_cols], test_data["target_volatility"]

        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        result = pd.DataFrame({
            "actual": y_test,
            "predicted": predictions,
            "naive_baseline": test_data["trailing_vol_21d"],
        }, index=test_data.index)

        all_predictions.append(result)

    return pd.concat(all_predictions)



def forecast_all_asset_volatilities(simple_returns: pd.DataFrame, train_end_date: str, target_horizon: int = 21) -> pd.Series:
    """
    Train a separate Linear Regression volatility forecasting model per
    asset, using data up to train_end_date only, and return each asset's
    forecasted next-period volatility as of that date.

    Parameters
    ----------
    simple_returns : pd.DataFrame
    train_end_date : str
    target_horizon : int, default 21

    Returns
    -------
    pd.Series
        Forecasted volatility per ticker.
    """
    feature_cols = ["trailing_vol_21d", "trailing_vol_60d", "trailing_mean_abs_return_21d", "trailing_return_21d"]
    forecasts = {}

    for ticker in simple_returns.columns:
        feature_df = build_volatility_features(simple_returns[ticker], target_horizon)
        train_data = feature_df[feature_df.index < train_end_date]

        if len(train_data) < 100:  # not enough history to train reliably yet
            forecasts[ticker] = simple_returns[ticker].loc[:train_end_date].tail(21).std() * np.sqrt(252)
            continue

        model = LinearRegression()
        model.fit(train_data[feature_cols], train_data["target_volatility"])

        # Predict using the most recent available feature row as of train_end_date
        latest_features = feature_df[feature_df.index < train_end_date][feature_cols].iloc[[-1]]
        forecasts[ticker] = model.predict(latest_features)[0]

    return pd.Series(forecasts)



def ml_inverse_volatility_portfolio(forecasted_vols: pd.Series) -> pd.Series:
    """
    Construct an Inverse Volatility portfolio using ML-FORECASTED volatility
    instead of historical volatility.

    Parameters
    ----------
    forecasted_vols : pd.Series
        Forecasted volatility per ticker (output of
        forecast_all_asset_volatilities).

    Returns
    -------
    pd.Series
        Weights indexed by ticker, summing to 1.
    """
    inverse_vol = 1 / forecasted_vols
    return inverse_vol / inverse_vol.sum()


def walk_forward_ml_vs_historical(
    simple_returns: pd.DataFrame,
    initial_train_years: int = 10,
    transaction_cost: float = 0.001,
) -> dict:
    """
    Walk-forward comparison of ML-Enhanced vs Historical Inverse Volatility
    portfolios, re-estimating both at each annual rebalancing date using
    only data available up to that point.

    Parameters
    ----------
    simple_returns : pd.DataFrame
    initial_train_years : int, default 10
    transaction_cost : float, default 0.001

    Returns
    -------
    dict
        oos_returns for both "ML-Enhanced" and "Historical" as pd.Series,
        plus weights_history for both.
    """
    start_date = simple_returns.index.min()
    first_rebalance = start_date + pd.DateOffset(years=initial_train_years)
    rebalance_dates = pd.date_range(start=first_rebalance, end=simple_returns.index.max(), freq="YS")

    ml_returns_list, hist_returns_list = [], []
    ml_weights_history, hist_weights_history = {}, {}

    for i, rebal_date in enumerate(rebalance_dates):
        train_window = simple_returns[simple_returns.index < rebal_date]
        next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else simple_returns.index.max() + pd.Timedelta(days=1)
        test_window = simple_returns[(simple_returns.index >= rebal_date) & (simple_returns.index < next_date)]

        if len(test_window) == 0:
            continue

        # ML-Enhanced
        forecasted_vols = forecast_all_asset_volatilities(simple_returns, train_end_date=str(rebal_date.date()))
        ml_weights = ml_inverse_volatility_portfolio(forecasted_vols)
        ml_weights_history[str(rebal_date.date())] = ml_weights
        ml_period_returns = test_window.dot(ml_weights.values)
        ml_period_returns.iloc[0] -= transaction_cost
        ml_returns_list.append(ml_period_returns)

        # Historical baseline
        hist_weights = inverse_volatility_portfolio(train_window)
        hist_weights_history[str(rebal_date.date())] = hist_weights
        hist_period_returns = test_window.dot(hist_weights.values)
        hist_period_returns.iloc[0] -= transaction_cost
        hist_returns_list.append(hist_period_returns)

    return {
        "ML-Enhanced": {"oos_returns": pd.concat(ml_returns_list), "weights_history": ml_weights_history},
        "Historical": {"oos_returns": pd.concat(hist_returns_list), "weights_history": hist_weights_history},
    }