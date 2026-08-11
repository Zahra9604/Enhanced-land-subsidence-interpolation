# ============================================================
# Evaluation and Prediction
# ============================================================
# This script evaluates the performance of a trained model on the training, validation, and test datasets. It calculates various metrics, generates predictions, and saves the results to CSV files. Additionally, it plots training and validation loss and MAE curves, as well as observed vs predicted values.
# Requires Libraries

import os
import csv
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from utils import load_coordinates, save_predictions, calculate_metrics
from sklearn.preprocessing import MinMaxScaler
# ============================================================
# Paths
# ============================================================

current_dir = os.getcwd()
root = os.path.dirname(current_dir)

dataset_path = os.path.join(root, "data")
output_path = os.path.join(root, "results")

os.makedirs(output_path, exist_ok=True)

print("Dataset path:", dataset_path)
print("Output path:", output_path)


# ============================================================
# Load saved model
# ============================================================

model_path = os.path.join(
    dataset_path,
    "Bestmodel.keras"
)

print("\nLoading model...")

model = tf.keras.models.load_model(model_path)

print("Model loaded successfully.")
model.summary()


# ============================================================
# Load datasets for evaluation
# ============================================================

datasets = {}

for split in ["train", "val", "test"]:

    datasets[split] = {
        "x": np.load(
            os.path.join(
                dataset_path,
                f"{split}_data.npy"
            )
        ),

        "y": np.load(
            os.path.join(
                dataset_path,
                f"{split}_label.npy"
            )
        ),
            "y_real": np.load(os.path.join(dataset_path, f"{split}_label_real.npy"))} 


x_train = datasets["train"]["x"]
y_train = datasets["train"]["y"]
y_train_real = datasets["train"]["y_real"]

x_val = datasets["val"]["x"]
y_val = datasets["val"]["y"]
y_val_real = datasets["val"]["y_real"]
x_test = datasets["test"]["x"]
y_test = datasets["test"]["y"]
y_test_real = datasets["test"]["y_real"]

print("\nDataset shapes:")

print(f"x_train: {x_train.shape}")
print(f"y_train: {y_train.shape}")

print(f"x_val:   {x_val.shape}")
print(f"y_val:   {y_val.shape}")

print(f"x_test:  {x_test.shape}")
print(f"y_test:  {y_test.shape}")


# ============================================================
#  Evaluate model
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


train_results = model.evaluate(
    x_train,
    y_train,
    verbose=0
)

val_results = model.evaluate(
    x_val,
    y_val,
    verbose=0
)

test_results = model.evaluate(
    x_test,
    y_test,
    verbose=0
)


train_loss = float(train_results[0])
train_mae = float(train_results[1])

val_loss = float(val_results[0])
val_mae = float(val_results[1])

test_loss = float(test_results[0])
test_mae = float(test_results[1])


print("\nScaled-data performance:")

print(
    f"Train      MSE: {train_loss:.6f}   "
    f"MAE: {train_mae:.6f}"
)

print(
    f"Validation MSE: {val_loss:.6f}   "
    f"MAE: {val_mae:.6f}"
)

print(
    f"Test       MSE: {test_loss:.6f}   "
    f"MAE: {test_mae:.6f}"
)


# ============================================================
# Generate predictions
# ============================================================

pred_train = model.predict(x_train, verbose=1)
pred_val   = model.predict(x_val, verbose=1)
pred_test  = model.predict(x_test, verbose=1)



# ============================================================
# Convert predictions and targets back to original units
# ============================================================
pred_train_real = pred_train.flatten() * 10.0
pred_val_real   = pred_val.flatten() * 10.0
pred_test_real  = pred_test.flatten() * 10.0

y_train_real = y_train.flatten() * 10.0
y_val_real   = y_val.flatten() * 10.0
y_test_real  = y_test.flatten() * 10.0


# ============================================================
# Calculate metrics
# ============================================================

train_mse, train_rmse, train_mae_real, train_r2 = calculate_metrics(
    y_train_real,
    pred_train_real
)

val_mse, val_rmse, val_mae_real, val_r2 = calculate_metrics(
    y_val_real,
    pred_val_real
)

test_mse, test_rmse, test_mae_real, test_r2 = calculate_metrics(
    y_test_real,
    pred_test_real
)


# ============================================================
# Print results
# ============================================================

print("\n================ Model Performance ================")

print("\nTraining:")
print(f"  MSE:  {train_mse:.4f}")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  MAE:  {train_mae_real:.4f}")
print(f"  R²:   {train_r2:.4f}")

print("\nValidation:")
print(f"  MSE:  {val_mse:.4f}")
print(f"  RMSE: {val_rmse:.4f}")
print(f"  MAE:  {val_mae_real:.4f}")
print(f"  R²:   {val_r2:.4f}")

print("\nTest:")
print(f"  MSE:  {test_mse:.4f}")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE:  {test_mae_real:.4f}")
print(f"  R²:   {test_r2:.4f}")
# ============================================================
# Print final results
# ============================================================

print("\n" + "=" * 60)
print("PERFORMANCE IN ORIGINAL UNITS")
print("=" * 60)

print("\nTraining:")
print(f"MSE  : {train_mse:.6f}")
print(f"RMSE : {train_rmse:.6f}")
print(f"MAE  : {train_mae_real:.6f}")
print(f"R²   : {train_r2:.6f}")

print("\nValidation:")
print(f"MSE  : {val_mse:.6f}")
print(f"RMSE : {val_rmse:.6f}")
print(f"MAE  : {val_mae_real:.6f}")
print(f"R²   : {val_r2:.6f}")

print("\nTesting:")
print(f"MSE  : {test_mse:.6f}")
print(f"RMSE : {test_rmse:.6f}")
print(f"MAE  : {test_mae_real:.6f}")
print(f"R²   : {test_r2:.6f}")


# ============================================================
# 9. Save metrics
# ============================================================

metrics_file = os.path.join(
    output_path,
    "model_metrics.csv"
)

with open(
    metrics_file,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Dataset",
        "MSE",
        "RMSE",
        "MAE",
        "R2"
    ])

    writer.writerow([
        "Train",
        train_mse,
        train_rmse,
        train_mae_real,
        train_r2
    ])

    writer.writerow([
        "Validation",
        val_mse,
        val_rmse,
        val_mae_real,
        val_r2
    ])

    writer.writerow([
        "Test",
        test_mse,
        test_rmse,
        test_mae_real,
        test_r2
    ])

print(
    f"\nMetrics saved to: {metrics_file}"
)


# ============================================================
#  Load PS coordinates
# ============================================================

utmx_train, utmy_train = load_coordinates("train",dataset_path)
utmx_val, utmy_val = load_coordinates("val",dataset_path)
utmx_test, utmy_test = load_coordinates("test",dataset_path)


# ============================================================
# 11. Save predictions
# ============================================================


# Training predictions

save_predictions(
    os.path.join(
        output_path,
        "train_predictions.csv"
    ),
    utmx_train,
    utmy_train,
    pred_train_real,
    y_train_real
)


# Validation predictions

save_predictions(
    os.path.join(
        output_path,
        "validation_predictions.csv"
    ),
    utmx_val,
    utmy_val,
    pred_val_real,
    y_val_real
)


# Test predictions

save_predictions(
    os.path.join(
        output_path,
        "test_predictions.csv"
    ),
    utmx_test,
    utmy_test,
    pred_test_real,
    y_test_real
)


print(
    "\nPrediction CSV files saved."
)


# ============================================================
# 12. Plot training and validation loss
# ============================================================

history_path = os.path.join(
    dataset_path,
    "training_history.json"
)

if os.path.exists(history_path):

    with open(history_path, "r") as file:
        history = json.load(file)

    epochs = range(
        1,
        len(history["loss"]) + 1
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_plot_path = os.path.join(
        output_path,
        "training_validation_loss.png"
    )

    plt.savefig(
        loss_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Loss plot saved to: {loss_plot_path}"
    )


    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["mae"],
        label="Training MAE"
    )

    plt.plot(
        epochs,
        history["val_mae"],
        label="Validation MAE"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MAE")

    plt.title(
        "Training and Validation MAE"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    mae_plot_path = os.path.join(
        output_path,
        "training_validation_mae.png"
    )

    plt.savefig(
        mae_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"MAE plot saved to: {mae_plot_path}"
    )

else:

    print(
        "\nWARNING: training_history.json not found."
    )

    print(
        "Loss and MAE curves cannot be reconstructed "
        "from the saved .keras model alone."
    )


# ============================================================
# 13. Plot predicted vs observed
# ============================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test_real,
    pred_test_real,
    s=10,
    alpha=0.5
)

min_value = min(
    y_test_real.min(),
    pred_test_real.min()
)

max_value = max(
    y_test_real.max(),
    pred_test_real.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Observed Subsidence")
plt.ylabel("Predicted Subsidence")

plt.title(
    "Observed vs Predicted Subsidence"
)

plt.grid(True)
plt.tight_layout()

prediction_plot_path = os.path.join(
    output_path,
    "observed_vs_predicted_test.png"
)

plt.savefig(
    prediction_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Prediction plot saved to: "
    f"{prediction_plot_path}"
)


# ============================================================
# 14. Save model architecture
# ============================================================

architecture_path = os.path.join(
    output_path,
    "model_architecture.png"
)

tf.keras.utils.plot_model(
    model,
    to_file=architecture_path,
    show_shapes=True,
    show_layer_names=True,
    dpi=200
)

print(
    f"Model architecture saved to: "
    f"{architecture_path}"
)


# ============================================================
# 15. Finished
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)