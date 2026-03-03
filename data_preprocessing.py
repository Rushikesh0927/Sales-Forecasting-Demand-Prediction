"""
Data Preprocessing Module
=========================
Loads, cleans, and merges the Walmart Store Sales Forecasting dataset.
Handles missing values, date parsing, and data type conversions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOT_DIR = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def load_raw_data():
    """Load the four raw CSV files."""
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    features = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    stores = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
    return train, test, features, stores


def clean_and_merge(train, features, stores):
    """
    Merge train with features and stores, clean missing values,
    and parse dates.
    """
    train["Date"] = pd.to_datetime(train["Date"])
    features["Date"] = pd.to_datetime(features["Date"])

    # Merge: train + features (on Store, Date, IsHoliday)
    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")

    # Merge: + stores (on Store)
    df = df.merge(stores, on="Store", how="left")

    # Fill missing MarkDown columns with 0
    markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    for col in markdown_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Fill any remaining NAs in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Sort by Store, Dept, Date
    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    return df


def generate_eda_plots(df):
    """Generate and save EDA visualisation charts."""
    sns.set_theme(style="darkgrid", palette="viridis")

    # 1. Weekly Sales Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["Weekly_Sales"], bins=80, kde=True, ax=ax, color="#6C63FF")
    ax.set_title("Distribution of Weekly Sales", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Weekly Sales ($)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "sales_distribution.png"), dpi=120)
    plt.close(fig)

    # 2. Average Weekly Sales Over Time
    weekly_avg = df.groupby("Date")["Weekly_Sales"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(weekly_avg["Date"], weekly_avg["Weekly_Sales"], color="#FF6584", linewidth=1.5)
    ax.fill_between(weekly_avg["Date"], weekly_avg["Weekly_Sales"], alpha=0.15, color="#FF6584")
    ax.set_title("Average Weekly Sales Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg Weekly Sales ($)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "sales_trend.png"), dpi=120)
    plt.close(fig)

    # 3. Holiday vs Non-Holiday Sales
    fig, ax = plt.subplots(figsize=(7, 5))
    holiday_sales = df.groupby("IsHoliday")["Weekly_Sales"].mean()
    colors = ["#43AA8B", "#F94144"]
    holiday_sales.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Holiday vs Non-Holiday Avg Sales", fontsize=14, fontweight="bold")
    ax.set_xticklabels(["Non-Holiday", "Holiday"], rotation=0)
    ax.set_ylabel("Avg Weekly Sales ($)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "holiday_impact.png"), dpi=120)
    plt.close(fig)

    # 4. Sales by Store Type
    fig, ax = plt.subplots(figsize=(7, 5))
    type_sales = df.groupby("Type")["Weekly_Sales"].mean().sort_values(ascending=False)
    type_sales.plot(kind="bar", ax=ax, color=["#6C63FF", "#FF6584", "#43AA8B"], edgecolor="white")
    ax.set_title("Avg Weekly Sales by Store Type", fontsize=14, fontweight="bold")
    ax.set_ylabel("Avg Weekly Sales ($)")
    ax.set_xlabel("Store Type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "sales_by_store_type.png"), dpi=120)
    plt.close(fig)

    # 5. Top 10 Stores by Total Sales
    fig, ax = plt.subplots(figsize=(10, 5))
    top_stores = df.groupby("Store")["Weekly_Sales"].sum().nlargest(10)
    top_stores.plot(kind="barh", ax=ax, color="#6C63FF", edgecolor="white")
    ax.set_title("Top 10 Stores by Total Sales", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Sales ($)")
    ax.set_ylabel("Store")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "top_stores.png"), dpi=120)
    plt.close(fig)

    # 6. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_cols = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI",
                 "Unemployment", "Size", "MarkDown1", "MarkDown2",
                 "MarkDown3", "MarkDown4", "MarkDown5"]
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close(fig)

    print("  [OK] 6 EDA plots saved to %s" % PLOT_DIR)


def preprocess_pipeline():
    """Run the full preprocessing pipeline and return the cleaned DataFrame."""
    print(">> Loading raw data...")
    train, test, features, stores = load_raw_data()
    print("  [OK] train: %s, test: %s, features: %s, stores: %s" % (train.shape, test.shape, features.shape, stores.shape))

    print(">> Cleaning & merging...")
    df = clean_and_merge(train, features, stores)
    print("  [OK] Merged dataset: %s" % str(df.shape))
    print("  [OK] Date range: %s -> %s" % (df["Date"].min().date(), df["Date"].max().date()))
    print("  [OK] Stores: %d, Departments: %d" % (df["Store"].nunique(), df["Dept"].nunique()))

    print(">> Generating EDA plots...")
    generate_eda_plots(df)

    return df


if __name__ == "__main__":
    df = preprocess_pipeline()
    print("\n[DONE] Preprocessing complete! Shape: %s" % str(df.shape))
    print(df.head())
