"""
Model Training Module
======================
Trains Linear Regression, Random Forest, and ARIMA models
on the Walmart Store Sales dataset.
Saves trained models as .pkl files.
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def time_based_split(df, test_ratio=0.2):
    """
    Split data based on time -- last test_ratio of dates go to validation.
    This prevents data leakage from future to past.
    """
    dates = sorted(df["Date"].unique())
    split_idx = int(len(dates) * (1 - test_ratio))
    split_date = dates[split_idx]

    train = df[df["Date"] < split_date].copy()
    val = df[df["Date"] >= split_date].copy()

    print("  [OK] Train: %d rows (up to %s)" % (len(train), pd.Timestamp(split_date).date()))
    print("  [OK] Val:   %d rows (from %s)" % (len(val), pd.Timestamp(split_date).date()))

    return train, val


def train_linear_regression(X_train, y_train, X_val, y_val):
    """Train a Linear Regression model."""
    print("\n>> Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)

    model_path = os.path.join(MODEL_DIR, "linear_regression.pkl")
    joblib.dump(model, model_path)
    print("  [OK] Model saved to %s" % model_path)

    return model, val_pred


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train a Random Forest model with tuned hyperparameters."""
    print("\n>> Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)

    model_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    joblib.dump(model, model_path)
    print("  [OK] Model saved to %s" % model_path)

    # Feature importance
    feature_imp = pd.Series(model.feature_importances_, index=X_train.columns)
    top_features = feature_imp.nlargest(10)
    print("  [OK] Top 10 features:")
    for feat, imp in top_features.items():
        print("      %s: %.4f" % (feat, imp))

    return model, val_pred


def train_arima_model(df):
    """
    Train an ARIMA model on aggregated weekly sales.
    Returns forecast for validation period.
    """
    print("\n>> Training ARIMA model...")
    try:
        from statsmodels.tsa.arima.model import ARIMA

        # Aggregate to weekly total sales
        weekly = df.groupby("Date")["Weekly_Sales"].mean().sort_index()
        weekly.index = pd.DatetimeIndex(weekly.index, freq="W-FRI")

        # Split
        split_idx = int(len(weekly) * 0.8)
        train_ts = weekly.iloc[:split_idx]
        val_ts = weekly.iloc[split_idx:]

        # Fit ARIMA(2,1,2)
        model = ARIMA(train_ts, order=(2, 1, 2))
        fitted = model.fit()

        # Forecast
        forecast = fitted.forecast(steps=len(val_ts))

        # Save
        model_path = os.path.join(MODEL_DIR, "arima_model.pkl")
        joblib.dump(fitted, model_path)
        print("  [OK] ARIMA model saved to %s" % model_path)
        print("  [OK] Train period: %s -> %s" % (train_ts.index[0].date(), train_ts.index[-1].date()))
        print("  [OK] Forecast period: %s -> %s" % (val_ts.index[0].date(), val_ts.index[-1].date()))

        return fitted, val_ts.values, forecast.values

    except Exception as e:
        print("  [WARN] ARIMA training failed: %s" % str(e))
        return None, None, None


def train_all_models(df, feature_columns):
    """
    Train all models and return predictions for evaluation.
    """
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    # Time-based split
    print("\n>> Splitting data (time-based, 80/20)...")
    train_df, val_df = time_based_split(df)

    # Prepare X, y
    available_features = [c for c in feature_columns if c in df.columns]
    X_train = train_df[available_features]
    y_train = train_df["Weekly_Sales"]
    X_val = val_df[available_features]
    y_val = val_df["Weekly_Sales"]

    results = {}

    # 1. Linear Regression
    lr_model, lr_pred = train_linear_regression(X_train, y_train, X_val, y_val)
    results["Linear Regression"] = {"y_true": y_val.values, "y_pred": lr_pred}

    # 2. Random Forest
    rf_model, rf_pred = train_random_forest(X_train, y_train, X_val, y_val)
    results["Random Forest"] = {"y_true": y_val.values, "y_pred": rf_pred}

    # 3. ARIMA (on aggregated data)
    arima_model, arima_true, arima_pred = train_arima_model(df)
    if arima_true is not None:
        results["ARIMA"] = {"y_true": arima_true, "y_pred": arima_pred}

    # Save feature columns list
    meta = {"feature_columns": available_features}
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[DONE] All models trained and saved to %s" % MODEL_DIR)
    return results


if __name__ == "__main__":
    from data_preprocessing import preprocess_pipeline
    from feature_engineering import prepare_features, get_feature_columns

    df = preprocess_pipeline()
    df = prepare_features(df)
    results = train_all_models(df, get_feature_columns())
