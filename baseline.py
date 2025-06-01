import numpy as np
from datasets import load_dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from train import (
    compute_wlaes,
    load_config,
)


def main():
    config = load_config()

    # --- 1. Load Dataset ---
    print(f"Loading dataset from path: {config['dataset_path']}")
    raw_dataset = load_dataset(
        config["dataset_path"],
        name=config.get("dataset_subset"),
        split=config.get("dataset_split"),
    )
    print(f"Raw dataset loaded with {len(raw_dataset)} samples.")

    # --- 2. Filter out video posts (note_type == 2) ---
    dataset_non_video = raw_dataset.filter(lambda x: float(x["note_type"]) == 1.0)
    print(f"Dataset after filtering video posts: {len(dataset_non_video)} samples.")

    if len(dataset_non_video) == 0:
        print("No non-video posts found. Exiting.")
        return

    # --- 3. Compute WLAES ---
    wlaes_cfg = config["wlaes_weights"]
    dataset_with_wlaes = dataset_non_video.map(
        compute_wlaes,
        batched=True,
        fn_kwargs={
            "w_like": wlaes_cfg["like"],
            "w_collect": wlaes_cfg["collect"],
            "w_comment": wlaes_cfg["comment"],
        },
    )
    print("WLAES scores computed.")

    # --- 4. Split dataset consistent with train.py ---
    # This ensures the validation set is identical.
    # The `train_test_split` method from `datasets` is used.
    print("Splitting dataset into train and validation sets for baseline...")
    train_test_split = dataset_with_wlaes.train_test_split(
        test_size=config["training"]["val_size"], seed=config["training"]["seed"]
    )

    train_baseline_dataset = train_test_split["train"]
    eval_baseline_dataset = train_test_split["test"]

    print(f"Baseline training set size: {len(train_baseline_dataset)}")
    print(f"Baseline evaluation set size: {len(eval_baseline_dataset)}")

    # --- 5. "Train" the baseline model (Calculate mean WLAES from training data) ---
    train_wlaes_scores = np.array(train_baseline_dataset["labels"])
    if len(train_wlaes_scores) == 0:
        print("Baseline training set is empty. Cannot compute mean. Exiting.")
        return

    mean_wlaes_train = np.mean(train_wlaes_scores)
    median_wlaes_train = np.median(
        train_wlaes_scores
    )  # Also calculate median as a robust alternative

    print(f"\n--- Baseline Model (Mean/Median Predictor) ---")
    print(f"Mean WLAES on Training Set: {mean_wlaes_train:.4f}")
    print(f"Median WLAES on Training Set: {median_wlaes_train:.4f}")

    # --- 6. "Predict" on the validation set ---
    # For the mean predictor, all predictions are simply the mean_wlaes_train.
    # For the median predictor, all predictions are the median_wlaes_train.
    actual_wlaes_eval = np.array(eval_baseline_dataset["labels"])
    if len(actual_wlaes_eval) == 0:
        print("Baseline evaluation set is empty. Cannot evaluate. Exiting.")
        return

    predictions_mean = np.full_like(actual_wlaes_eval, fill_value=mean_wlaes_train)
    predictions_median = np.full_like(actual_wlaes_eval, fill_value=median_wlaes_train)

    # --- 7. Evaluate the baseline ---
    print("\n--- Evaluation of Mean Predictor Baseline on Validation Set ---")
    mse_mean = mean_squared_error(actual_wlaes_eval, predictions_mean)
    mae_mean = mean_absolute_error(actual_wlaes_eval, predictions_mean)
    r2_mean = r2_score(actual_wlaes_eval, predictions_mean)

    print(f"  Mean Squared Error (MSE): {mse_mean:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae_mean:.4f}")
    print(f"  R-squared (R2):          {r2_mean:.4f}")
    # RMSE can be useful for direct comparison with MAE
    print(f"  Root Mean Squared Error (RMSE): {np.sqrt(mse_mean):.4f}")

    print("\n--- Evaluation of Median Predictor Baseline on Validation Set ---")
    mse_median = mean_squared_error(actual_wlaes_eval, predictions_median)
    mae_median = mean_absolute_error(actual_wlaes_eval, predictions_median)
    r2_median = r2_score(actual_wlaes_eval, predictions_median)

    print(f"  Mean Squared Error (MSE): {mse_median:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae_median:.4f}")
    print(f"  R-squared (R2):          {r2_median:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {np.sqrt(mse_median):.4f}")

    # Also print statistics of the actual WLAES scores on the validation set
    print("\n--- Statistics of Actual WLAES on Validation Set ---")
    print(f"  Mean:     {np.mean(actual_wlaes_eval):.4f}")
    print(f"  Median:   {np.median(actual_wlaes_eval):.4f}")
    print(f"  Std Dev:  {np.std(actual_wlaes_eval):.4f}")
    print(f"  Min:      {np.min(actual_wlaes_eval):.4f}")
    print(f"  Max:      {np.max(actual_wlaes_eval):.4f}")


if __name__ == "__main__":
    main()
