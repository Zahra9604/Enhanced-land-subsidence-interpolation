
# ============================================================
# Model and Traditional Interpolation Performance Comparison
# ============================================================
#
# Models:
#   1. CNN
#   2. IDW
#   3. Ordinary Kriging
#   4. RBF
#
# Datasets:
#   1. Training
#   2. Testing
#
# Output:
#
# MODEL_INTERPOLATION_PERFORMANCE_COMPARISON.csv
#
# ============================================================


# ============================================================
# 1. Required libraries
# ============================================================

import os
import numpy as np
import pandas as pd
from utils import calculate_metrics


# ============================================================
# 2. Paths
# ============================================================

root = os.path.dirname(os.getcwd())

result_path = os.path.join(
    root,
    "results"
)


train_file = os.path.join(
    result_path,
    "train_predictions_interpolation.csv"
)


test_file = os.path.join(
    result_path,
    "test_predictions_interpolation.csv"
)


# ============================================================
# 3. Check files
# ============================================================

if not os.path.exists(train_file):

    raise FileNotFoundError(
        f"\nTraining file not found:\n{train_file}"
    )


if not os.path.exists(test_file):

    raise FileNotFoundError(
        f"\nTesting file not found:\n{test_file}"
    )


# ============================================================
# 4. Load datasets
# ============================================================

print("\nLoading datasets...")


train_dataset = pd.read_csv(
    train_file
)


test_dataset = pd.read_csv(
    test_file
)


print(
    f"Training samples: {len(train_dataset):,}"
)


print(
    f"Testing samples : {len(test_dataset):,}"
)


# ============================================================
# 5. Required columns
# ============================================================

required_columns = [
    "Ground_truth",
    "Prediction",
    "IDW_Prediction",
    "Kriging_Prediction",
    "RBF_Prediction"
]


for column in required_columns:

    if column not in train_dataset.columns:

        raise ValueError(
            f"Column '{column}' is missing "
            f"from training dataset."
        )


    if column not in test_dataset.columns:

        raise ValueError(
            f"Column '{column}' is missing "
            f"from testing dataset."
        )


print(
    "\nAll required columns found."
)


# ============================================================
# 6. Metric function
# ============================================================

def get_metrics(
    true_values,
    predictions
):

    true_values = np.asarray(
        true_values,
        dtype=np.float64
    )


    predictions = np.asarray(
        predictions,
        dtype=np.float64
    )


    # --------------------------------------------------------
    # Remove NaN and infinite values
    # --------------------------------------------------------

    valid = (
        np.isfinite(true_values)
        &
        np.isfinite(predictions)
    )


    true_valid = (
        true_values[
            valid
        ]
    )


    prediction_valid = (
        predictions[
            valid
        ]
    )


    if len(true_valid) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0
        )


    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    mse, rmse, mae, r2 = (
        calculate_metrics(
            true_valid,
            prediction_valid
        )
    )


    return (
        mse,
        rmse,
        mae,
        r2,
        len(true_valid)
    )


# ============================================================
# 7. Model definitions
# ============================================================

models = {

    "CNN": "Prediction",

    "IDW": "IDW_Prediction",

    "Kriging": "Kriging_Prediction",

    "RBF": "RBF_Prediction"

}


# ============================================================
# 8. Calculate all performance metrics
# ============================================================

print("\n" + "=" * 90)
print("CALCULATING MODEL PERFORMANCE")
print("=" * 90)


results = []


# ============================================================
# TRAINING PERFORMANCE
# ============================================================

print("\nTraining performance:")


for model_name, prediction_column in models.items():

    true_values = (
        train_dataset[
            "Ground_truth"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    predictions = (
        train_dataset[
            prediction_column
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    (
        mse,
        rmse,
        mae,
        r2,
        n
    ) = get_metrics(
        true_values,
        predictions
    )


    results.append(
        {
            "Dataset": "Train",
            "Method": model_name,
            "N": n,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }
    )


    print(
        f"{model_name:10s} | "
        f"N = {n:,} | "
        f"MSE = {mse:.6f} | "
        f"RMSE = {rmse:.6f} | "
        f"MAE = {mae:.6f} | "
        f"R2 = {r2:.6f}"
    )


# ============================================================
# TESTING PERFORMANCE
# ============================================================

print("\nTesting performance:")


for model_name, prediction_column in models.items():

    true_values = (
        test_dataset[
            "Ground_truth"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    predictions = (
        test_dataset[
            prediction_column
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    (
        mse,
        rmse,
        mae,
        r2,
        n
    ) = get_metrics(
        true_values,
        predictions
    )


    results.append(
        {
            "Dataset": "Test",
            "Method": model_name,
            "N": n,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }
    )


    print(
        f"{model_name:10s} | "
        f"N = {n:,} | "
        f"MSE = {mse:.6f} | "
        f"RMSE = {rmse:.6f} | "
        f"MAE = {mae:.6f} | "
        f"R2 = {r2:.6f}"
    )


# ============================================================
# 9. Create one comparison DataFrame
# ============================================================

results = pd.DataFrame(
    results
)


# ============================================================
# 10. Round numerical values
# ============================================================

results[
    [
        "MSE",
        "RMSE",
        "MAE",
        "R2"
    ]
] = results[
    [
        "MSE",
        "RMSE",
        "MAE",
        "R2"
    ]
].round(6)


# ============================================================
# 11. Display final comparison
# ============================================================

print("\n" + "=" * 100)
print("MODEL / INTERPOLATION PERFORMANCE COMPARISON")
print("=" * 100)


print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 12. Save ONE CSV containing train + test performance
# ============================================================

output_file = os.path.join(
    result_path,
    "MODEL_INTERPOLATION_PERFORMANCE_COMPARISON.csv"
)


results.to_csv(
    output_file,
    index=False
)


# ============================================================
# 13. Print output path
# ============================================================

print("\n" + "=" * 100)
print("RESULTS SAVED")
print("=" * 100)

