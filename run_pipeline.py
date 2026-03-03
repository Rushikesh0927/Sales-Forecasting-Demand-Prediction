"""
Run Pipeline
=============
One-click script to run the entire pipeline:
  preprocess → feature engineering → train models → evaluate
"""

import time


def main():
    start = time.time()

    print("=" * 60)
    print("  WALMART SALES FORECASTING PIPELINE")
    print("=" * 60)

    # Step 1: Preprocessing
    from data_preprocessing import preprocess_pipeline
    df = preprocess_pipeline()

    # Step 2: Feature Engineering
    from feature_engineering import prepare_features, get_feature_columns
    df = prepare_features(df)

    # Step 3: Model Training
    from model_training import train_all_models
    results = train_all_models(df, get_feature_columns())

    # Step 4: Evaluation
    from evaluate import evaluate_all_models
    eval_results = evaluate_all_models(results)

    # Step 5: Compute Historical Averages
    from compute_averages import compute_averages
    compute_averages()

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"\n  To start the dashboard:")
    print(f"    python app.py")
    print(f"    Open http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
