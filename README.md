# Sales Forecasting & Demand Prediction System

A machine learning project that predicts future **department-level weekly sales** for Walmart stores using historical sales data. Built with Python, Scikit-learn, and Flask.

> **Dataset**: [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) (Kaggle)

---

## Project Overview

| Metric | Value |
|--------|-------|
| **Stores** | 45 (Type A, B, C) |
| **Departments** | 81 |
| **Training Data** | 421,570 rows (Feb 2010 - Oct 2012) |
| **Features** | 30 engineered features |
| **Models** | Linear Regression, Random Forest, ARIMA |
| **Deployment** | Flask API + Interactive Dashboard |

---

## Architecture

```
walmart-sales-forecasting/
|
|-- data/                        # Raw CSV data files
|   |-- train.csv                  421K rows - weekly sales per store-dept
|   |-- test.csv                   115K rows - prediction targets
|   |-- features.csv               External features (temp, CPI, fuel, markdowns)
|   |-- stores.csv                 Store metadata (type, size)
|
|-- models/                      # Trained models & metadata
|   |-- random_forest.pkl          Best performing ML model
|   |-- linear_regression.pkl      Baseline model
|   |-- arima_model.pkl            Time-series model
|   |-- historical_averages.json   Per store-dept averages for predictions
|   |-- evaluation_results.json    MAE & RMSE comparison
|   |-- model_meta.json            Feature column list
|
|-- static/plots/                # Auto-generated EDA charts (6 PNGs)
|-- templates/dashboard.html     # Flask dashboard UI
|
|-- data_preprocessing.py        # Load, clean, merge, generate EDA plots
|-- feature_engineering.py       # Lag, rolling, seasonal, holiday features
|-- model_training.py            # Train LR, RF, ARIMA with time-based split
|-- evaluate.py                  # MAE & RMSE metrics + comparison chart
|-- compute_averages.py          # Precompute historical averages
|-- app.py                       # Flask API + dashboard (port 5050)
|-- run_pipeline.py              # One-click full pipeline
|-- requirements.txt             # Python dependencies
```

---

## Model Results

| Model | MAE ($) | RMSE ($) | Notes |
|-------|---------|----------|-------|
| **ARIMA** | 469.76 | 604.10 | Best on aggregated store-level data |
| **Random Forest** | 1,398.04 | 2,885.26 | Best for dept-level predictions (used in dashboard) |
| Linear Regression | 1,670.45 | 3,325.20 | Baseline comparison |

**Key insight**: Random Forest captures cross-feature interactions (store size, department, markdowns) that ARIMA cannot. ARIMA excels at store-level time-series patterns.

### Top Predictive Features (Random Forest)
1. `Sales_Lag_1` (25.2%) - Last week's sales
2. `Sales_RollingMean_4` (20.0%) - 4-week moving average
3. `Sales_Lag_2` (17.9%) - Sales from 2 weeks ago
4. `Sales_RollingMean_12` (16.3%) - 12-week moving average
5. `Sales_Lag_4` (12.9%) - Sales from 4 weeks ago

---

## Feature Engineering

| Feature Category | Features | Purpose |
|-----------------|----------|---------|
| **Lag Features** | Sales_Lag_1, 2, 4 | Capture recent sales trends |
| **Rolling Mean** | RollingMean_4, 12 | Smooth out weekly noise |
| **Seasonal** | Month, Week, Quarter, DayOfYear | Capture seasonal patterns |
| **Holiday** | Is_SuperBowl, Thanksgiving, Christmas, LaborDay | Sales spikes during holidays |
| **Store** | StoreType_A/B/C, Size | Store-level characteristics |
| **Economic** | CPI, Unemployment, Fuel_Price, MarkDown1-5 | External economic factors |

---

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation & Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (preprocess + train + evaluate)
python run_pipeline.py

# Start the dashboard
python app.py
# Open http://127.0.0.1:5050
```

### API Usage
```bash
# GET prediction via REST API
curl "http://127.0.0.1:5050/api/predict?store=1&dept=1&date=2012-12-01"

# Response:
{
  "store": 1,
  "dept": 1,
  "date": "2012-12-01",
  "predicted_weekly_sales": 16633.38,
  "confidence": "High",
  "model": "Random Forest"
}
```

---

## Dashboard Features

- **Stats Overview**: Store count, department count, training data size
- **Model Comparison**: Side-by-side MAE/RMSE table with best model badge
- **Interactive Prediction**: Dropdown selectors for Store (with type/size) and Department
- **Confidence Indicator**: Shows prediction reliability based on historical data availability
- **EDA Visualizations**: 6 interactive charts (click to zoom): sales distribution, trends, holiday impact, store types, top stores, correlation heatmap
- **REST API**: Programmatic access for integration
- **Prediction Limitations**: Transparent disclosure of model limitations

---

## Prediction Limitations

1. **Date Range**: Training data covers Feb 2010 - Oct 2012. Predictions for dates far outside this range are less accurate.
2. **Lag Features**: Real-time predictions use historical averages as proxies. In production, recent actual sales would be used.
3. **Granularity**: Predictions are at department level, not individual product level.
4. **External Factors**: Local events, extreme weather, promotions, and pandemic effects are not modeled.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, Statsmodels |
| Visualization | Matplotlib, Seaborn |
| Web Framework | Flask |
| Model Persistence | Joblib |

---

## Author

Built as a Sales Forecasting & Demand Prediction project using the Walmart Store Sales Forecasting dataset from Kaggle.
