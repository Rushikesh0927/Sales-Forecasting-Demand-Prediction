"""
Feature Engineering Module
===========================
Creates lag features, rolling means, seasonal indicators, and
encodes categorical variables for the Walmart sales dataset.
"""

import pandas as pd
import numpy as np


def add_time_features(df):
    """Extract useful time components from the Date column."""
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfYear"] = df["Date"].dt.dayofyear
    return df


def add_lag_features(df, lags=(1, 2, 4)):
    """
    Create lag features for Weekly_Sales.
    Lags are computed per Store-Dept group.
    """
    df = df.copy()
    for lag in lags:
        df["Sales_Lag_%d" % lag] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
            .shift(lag)
        )
    return df


def add_rolling_features(df, windows=(4, 12)):
    """
    Create rolling mean features for Weekly_Sales.
    Rolling windows are computed per Store-Dept group.
    """
    df = df.copy()
    for w in windows:
        df["Sales_RollingMean_%d" % w] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
            .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
        )
    return df


def encode_store_type(df):
    """One-hot encode the Store Type column (A, B, C)."""
    df = df.copy()
    type_dummies = pd.get_dummies(df["Type"], prefix="StoreType", dtype=int)
    df = pd.concat([df, type_dummies], axis=1)
    return df


def add_holiday_indicators(df):
    """
    Add specific holiday indicators based on known Walmart holiday weeks:
    - Super Bowl, Labor Day, Thanksgiving, Christmas
    """
    df = df.copy()
    df["IsHoliday_binary"] = df["IsHoliday"].astype(int)

    super_bowl_weeks = [6]
    labor_day_weeks = [36]
    thanksgiving_weeks = [47]
    christmas_weeks = [52, 1]

    df["Is_SuperBowl"] = df["Week"].isin(super_bowl_weeks).astype(int)
    df["Is_LaborDay"] = df["Week"].isin(labor_day_weeks).astype(int)
    df["Is_Thanksgiving"] = df["Week"].isin(thanksgiving_weeks).astype(int)
    df["Is_Christmas"] = df["Week"].isin(christmas_weeks).astype(int)

    return df


def prepare_features(df):
    """
    Run the full feature engineering pipeline.
    Returns the DataFrame with all engineered features.
    """
    print(">> Adding time features...")
    df = add_time_features(df)

    print(">> Adding lag features (1, 2, 4 weeks)...")
    df = add_lag_features(df)

    print(">> Adding rolling mean features (4, 12 weeks)...")
    df = add_rolling_features(df)

    print(">> Encoding store type...")
    df = encode_store_type(df)

    print(">> Adding holiday indicators...")
    df = add_holiday_indicators(df)

    # Drop rows with NaN from lag/rolling (first few weeks per group)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    print("  [OK] Dropped %d rows with NaN from lag/rolling features" % (before - after))
    print("  [OK] Final feature set: %d columns, %d rows" % (df.shape[1], df.shape[0]))

    return df


def get_feature_columns():
    """Return the list of feature column names used for modeling."""
    return [
        "Store", "Dept", "Temperature", "Fuel_Price",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
        "CPI", "Unemployment", "Size",
        "Year", "Month", "Week", "Quarter", "DayOfYear",
        "Sales_Lag_1", "Sales_Lag_2", "Sales_Lag_4",
        "Sales_RollingMean_4", "Sales_RollingMean_12",
        "StoreType_A", "StoreType_B", "StoreType_C",
        "IsHoliday_binary", "Is_SuperBowl", "Is_LaborDay",
        "Is_Thanksgiving", "Is_Christmas",
    ]


if __name__ == "__main__":
    from data_preprocessing import preprocess_pipeline

    df = preprocess_pipeline()
    df = prepare_features(df)
    print("\n[DONE] Feature engineering complete! Shape: %s" % str(df.shape))
    print("Columns:", list(df.columns))
