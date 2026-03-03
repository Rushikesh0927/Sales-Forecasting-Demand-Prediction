"""
Compute Historical Averages
=============================
Precomputes per-store-dept historical averages from the training data
and saves them as a JSON file for use in real-time predictions.
"""

import os
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def compute_averages():
    """Compute historical averages per Store-Dept and per Store."""
    print(">> Computing historical averages...")

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    features = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    stores = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))

    train["Date"] = pd.to_datetime(train["Date"])
    features["Date"] = pd.to_datetime(features["Date"])

    # Merge for full context
    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.merge(stores, on="Store", how="left")

    # Fill NAs in MarkDown columns
    for col in ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Per Store-Dept averages
    store_dept_avg = df.groupby(["Store", "Dept"]).agg({
        "Weekly_Sales": ["mean", "std", "min", "max", "count"],
        "Temperature": "mean",
        "Fuel_Price": "mean",
        "CPI": "mean",
        "Unemployment": "mean",
        "MarkDown1": "mean",
        "MarkDown2": "mean",
        "MarkDown3": "mean",
        "MarkDown4": "mean",
        "MarkDown5": "mean",
    }).reset_index()

    # Flatten multi-level columns
    store_dept_avg.columns = [
        "Store", "Dept",
        "avg_sales", "std_sales", "min_sales", "max_sales", "num_weeks",
        "avg_temp", "avg_fuel", "avg_cpi", "avg_unemployment",
        "avg_md1", "avg_md2", "avg_md3", "avg_md4", "avg_md5",
    ]

    # Convert to dict keyed by "store_dept"
    hist_averages = {}
    for _, row in store_dept_avg.iterrows():
        key = "%d_%d" % (int(row["Store"]), int(row["Dept"]))
        hist_averages[key] = {
            "avg_sales": round(float(row["avg_sales"]), 2),
            "std_sales": round(float(row["std_sales"]), 2) if not pd.isna(row["std_sales"]) else 0,
            "min_sales": round(float(row["min_sales"]), 2),
            "max_sales": round(float(row["max_sales"]), 2),
            "num_weeks": int(row["num_weeks"]),
            "avg_temp": round(float(row["avg_temp"]), 2),
            "avg_fuel": round(float(row["avg_fuel"]), 2),
            "avg_cpi": round(float(row["avg_cpi"]), 2),
            "avg_unemployment": round(float(row["avg_unemployment"]), 2),
            "avg_md1": round(float(row["avg_md1"]), 2),
            "avg_md2": round(float(row["avg_md2"]), 2),
            "avg_md3": round(float(row["avg_md3"]), 2),
            "avg_md4": round(float(row["avg_md4"]), 2),
            "avg_md5": round(float(row["avg_md5"]), 2),
        }

    # Also compute global averages as fallback
    global_avg = {
        "avg_sales": round(float(df["Weekly_Sales"].mean()), 2),
        "avg_temp": round(float(df["Temperature"].mean()), 2),
        "avg_fuel": round(float(df["Fuel_Price"].mean()), 2),
        "avg_cpi": round(float(df["CPI"].mean()), 2),
        "avg_unemployment": round(float(df["Unemployment"].mean()), 2),
        "avg_md1": round(float(df["MarkDown1"].mean()), 2),
        "avg_md2": round(float(df["MarkDown2"].mean()), 2),
        "avg_md3": round(float(df["MarkDown3"].mean()), 2),
        "avg_md4": round(float(df["MarkDown4"].mean()), 2),
        "avg_md5": round(float(df["MarkDown5"].mean()), 2),
    }

    # Date range info
    date_info = {
        "train_start": str(df["Date"].min().date()),
        "train_end": str(df["Date"].max().date()),
        "num_stores": int(df["Store"].nunique()),
        "num_depts": int(df["Dept"].nunique()),
        "total_rows": int(len(df)),
    }

    result = {
        "store_dept": hist_averages,
        "global": global_avg,
        "date_info": date_info,
    }

    output_path = os.path.join(MODEL_DIR, "historical_averages.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("  [OK] Saved %d store-dept averages to %s" % (len(hist_averages), output_path))
    print("  [OK] Global avg weekly sales: $%.2f" % global_avg["avg_sales"])

    return result


if __name__ == "__main__":
    compute_averages()
