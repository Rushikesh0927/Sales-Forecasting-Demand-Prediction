"""
Model Evaluation Module
========================
Computes MAE and RMSE for all trained models.
Generates comparison plots and saves results.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def mean_absolute_error(y_true, y_pred):
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true, y_pred):
    """Compute Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def evaluate_all_models(results):
    """
    Evaluate all models and print comparison table.
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    eval_results = {}

    print("\n%-25s %12s %12s" % ("Model", "MAE", "RMSE"))
    print("-" * 50)

    for name, data in results.items():
        y_true = np.array(data["y_true"])
        y_pred = np.array(data["y_pred"])

        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)

        eval_results[name] = {"MAE": round(float(mae), 2), "RMSE": round(float(rmse), 2)}
        print("%-25s %12.2f %12.2f" % (name, mae, rmse))

    print("-" * 50)

    # Find best model
    best_model = min(eval_results, key=lambda k: eval_results[k]["RMSE"])
    print("\n** Best model (lowest RMSE): %s" % best_model)

    # Save results
    results_path = os.path.join(MODEL_DIR, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print("  [OK] Results saved to %s" % results_path)

    # Generate comparison plot
    _plot_comparison(eval_results)

    return eval_results


def _plot_comparison(eval_results):
    """Generate a bar chart comparing model performance."""
    models = list(eval_results.keys())
    mae_values = [eval_results[m]["MAE"] for m in models]
    rmse_values = [eval_results[m]["RMSE"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, mae_values, width, label="MAE",
                   color="#6C63FF", edgecolor="white", alpha=0.9)
    bars2 = ax.bar(x + width / 2, rmse_values, width, label="RMSE",
                   color="#FF6584", edgecolor="white", alpha=0.9)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Error ($)", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate("$%s" % f"{height:,.0f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate("$%s" % f"{height:,.0f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "model_comparison.png"), dpi=120)
    plt.close(fig)
    print("  [OK] Comparison plot saved")


if __name__ == "__main__":
    from data_preprocessing import preprocess_pipeline
    from feature_engineering import prepare_features, get_feature_columns
    from model_training import train_all_models

    df = preprocess_pipeline()
    df = prepare_features(df)
    results = train_all_models(df, get_feature_columns())
    evaluate_all_models(results)
