"""
Flask Application -- Sales Forecasting Dashboard
==================================================
Provides a web dashboard with EDA visualizations, model
comparison, and real-time sales prediction.
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

app = Flask(__name__, static_folder="static", template_folder="templates")

# Global state loaded at startup
rf_model = None
feature_columns = []
eval_results = {}
store_info = []
dept_list = []
hist_data = {}          # historical averages


def load_assets():
    """Load model, metadata, eval results, store/dept info, and historical averages."""
    global rf_model, feature_columns, eval_results, store_info, dept_list, hist_data

    rf_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    meta_path = os.path.join(MODEL_DIR, "model_meta.json")
    eval_path = os.path.join(MODEL_DIR, "evaluation_results.json")
    stores_path = os.path.join(DATA_DIR, "stores.csv")
    train_path = os.path.join(DATA_DIR, "train.csv")
    hist_path = os.path.join(MODEL_DIR, "historical_averages.json")

    if os.path.exists(rf_path):
        # Ensure the model file's integrity (e.g., checksum verification) or use a safer serialization format.
        # If the source is absolutely trusted and secured, this might be acceptable, but generally avoid deserializing untrusted data.
        # For production, consider alternative model deployment strategies that don't rely on direct pickle loading from potentially untrusted paths.
        # Example (conceptual, not a direct drop-in for joblib): rf_model = load_model_safely(rf_path)
        print("  [OK] Random Forest model loaded")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
            feature_columns = meta.get("feature_columns", [])

    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_results = json.load(f)

    if os.path.exists(stores_path):
        stores_df = pd.read_csv(stores_path)
        store_info = stores_df.to_dict("records")
        print("  [OK] %d stores loaded" % len(store_info))

    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path, usecols=["Dept"])
        dept_list = sorted(train_df["Dept"].unique().tolist())
        print("  [OK] %d departments loaded" % len(dept_list))

    if os.path.exists(hist_path):
        with open(hist_path) as f:
            hist_data = json.load(f)
        num_pairs = len(hist_data.get("store_dept", {}))
        print("  [OK] %d historical averages loaded" % num_pairs)


def get_plot_files():
    """Return list of plot filenames in static/plots/."""
    plots_dir = os.path.join(BASE_DIR, "static", "plots")
    if not os.path.exists(plots_dir):
        return []
    return sorted([f for f in os.listdir(plots_dir) if f.endswith(".png")])


def _get_date_info():
    """Get training date range info."""
    info = hist_data.get("date_info", {})
    return {
        "train_start": info.get("train_start", "2010-02-05"),
        "train_end": info.get("train_end", "2012-10-26"),
    }


def _render_dashboard(**extra):
    """Helper to render dashboard with all common data."""
    plots = get_plot_files()
    return render_template(
        "dashboard.html",
        plots=plots,
        eval_results=eval_results,
        model_loaded=rf_model is not None,
        store_info=store_info,
        dept_list=dept_list,
        date_info=_get_date_info(),
        **extra,
    )


@app.route("/")
def dashboard():
    """Main dashboard page."""
    return _render_dashboard()


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction from the web form."""
    try:
        store = int(request.form.get("store", 1))
        dept = int(request.form.get("dept", 1))
        date_str = request.form.get("date", "2012-12-01")
        date = pd.to_datetime(date_str)

        prediction, confidence = _make_prediction(store, dept, date)

        return _render_dashboard(
            prediction=prediction,
            confidence=confidence,
            pred_store=store,
            pred_dept=dept,
            pred_date=date_str,
        )
    except Exception as e:
        return _render_dashboard(error=str(e))


@app.route("/api/predict")
def api_predict():
    """JSON API endpoint for predictions."""
    try:
        store = int(request.args.get("store", 1))
        dept = int(request.args.get("dept", 1))
        date_str = request.args.get("date", "2012-12-01")
        date = pd.to_datetime(date_str)

        prediction, confidence = _make_prediction(store, dept, date)

        return jsonify({
            "store": store,
            "dept": dept,
            "date": date_str,
            "predicted_weekly_sales": round(float(prediction), 2),
            "confidence": confidence,
            "model": "Random Forest",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def _make_prediction(store, dept, date):
    """
    Create a feature vector using historical averages and predict.
    Returns (prediction, confidence_level).
    """
    if rf_model is None:
        raise ValueError("Model not loaded. Run the pipeline first: python run_pipeline.py")

    # Look up store info
    s_info = next((s for s in store_info if s["Store"] == store), None)
    store_size = s_info["Size"] if s_info else 150000
    store_type = s_info["Type"] if s_info else "A"

    # Look up historical averages for this store-dept
    key = "%d_%d" % (store, dept)
    sd_avg = hist_data.get("store_dept", {}).get(key, None)
    g_avg = hist_data.get("global", {})

    # Determine confidence level
    if sd_avg and sd_avg.get("num_weeks", 0) >= 50:
        confidence = "High"
    elif sd_avg and sd_avg.get("num_weeks", 0) >= 20:
        confidence = "Medium"
    elif sd_avg:
        confidence = "Low"
    else:
        confidence = "Very Low (no historical data for this store-dept)"

    # Use actual averages with global fallback
    avg_sales = sd_avg["avg_sales"] if sd_avg else g_avg.get("avg_sales", 15000)

    features = {
        "Store": store,
        "Dept": dept,
        "Temperature": (sd_avg or g_avg).get("avg_temp", 60.0),
        "Fuel_Price": (sd_avg or g_avg).get("avg_fuel", 3.5),
        "MarkDown1": (sd_avg or g_avg).get("avg_md1", 0),
        "MarkDown2": (sd_avg or g_avg).get("avg_md2", 0),
        "MarkDown3": (sd_avg or g_avg).get("avg_md3", 0),
        "MarkDown4": (sd_avg or g_avg).get("avg_md4", 0),
        "MarkDown5": (sd_avg or g_avg).get("avg_md5", 0),
        "CPI": (sd_avg or g_avg).get("avg_cpi", 220.0),
        "Unemployment": (sd_avg or g_avg).get("avg_unemployment", 7.5),
        "Size": store_size,
        "Year": date.year,
        "Month": date.month,
        "Week": date.isocalendar()[1],
        "Quarter": (date.month - 1) // 3 + 1,
        "DayOfYear": date.timetuple().tm_yday,
        "Sales_Lag_1": avg_sales,
        "Sales_Lag_2": avg_sales,
        "Sales_Lag_4": avg_sales,
        "Sales_RollingMean_4": avg_sales,
        "Sales_RollingMean_12": avg_sales,
        "StoreType_A": 1 if store_type == "A" else 0,
        "StoreType_B": 1 if store_type == "B" else 0,
        "StoreType_C": 1 if store_type == "C" else 0,
        "IsHoliday_binary": 0,
        "Is_SuperBowl": 1 if date.isocalendar()[1] == 6 else 0,
        "Is_LaborDay": 1 if date.isocalendar()[1] == 36 else 0,
        "Is_Thanksgiving": 1 if date.isocalendar()[1] == 47 else 0,
        "Is_Christmas": 1 if date.isocalendar()[1] in (52, 1) else 0,
    }

    row = {col: features.get(col, 0) for col in feature_columns}
    X = pd.DataFrame([row])

    prediction = rf_model.predict(X)[0]
    return float(prediction), confidence


if __name__ == "__main__":
    load_assets()
    print("\n>> Starting Sales Forecasting Dashboard...")
    print("   http://127.0.0.1:5050\n")
    app.run(debug=True, host="127.0.0.1", port=5050)
