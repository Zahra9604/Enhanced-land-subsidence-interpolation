# Required libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
# ============================================================
# Dataset path
# ============================================================

root = os.path.dirname(os.getcwd())
dataset_path = os.path.join(root, "data")


# ============================================================
# Load train / validation / test data
# ============================================================

datasets = {}

for split in ["train", "val", "test"]:

    datasets[split] = {
        "x": np.load(os.path.join(dataset_path, f"{split}_data.npy")),
        "y": np.load(os.path.join(dataset_path, f"{split}_label.npy")),
        "indices": np.load(
            os.path.join(dataset_path, f"ind_{split}.npy")
        ),
    }


# ============================================================
# Access datasets
# ============================================================

x_train = datasets["train"]["x"]
y_train = datasets["train"]["y"]
ind_train = datasets["train"]["indices"]

x_val = datasets["val"]["x"]
y_val = datasets["val"]["y"]
ind_val = datasets["val"]["indices"]

x_test = datasets["test"]["x"]
y_test = datasets["test"]["y"]
ind_test = datasets["test"]["indices"]


# Define the convolutional neural network architecture
model = Sequential([
    Conv2D(
        32,
        (1, 1),
        padding="same",
        activation="relu",
        input_shape=(30, 30, 9)
    ),
    Conv2D(32, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.1),

    Conv2D(32, (1, 1), padding="same", activation="relu"),
    Conv2D(32, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.1),

    Conv2D(1024, (1, 1), padding="same", activation="relu"),
    Conv2D(1024, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.1),

    Flatten(),

    Dense(1024, activation="relu"),
    BatchNormalization(),
    Dropout(0.1),

    Dense(1024, activation="relu"),
    BatchNormalization(),
    Dropout(0.1),

    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.1),

    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.1),

    Dense(256, activation="relu"),
    BatchNormalization(),

    Dense(1, activation="linear")
])
model.summary()


#model compile
model.compile(
    loss="mse",
    optimizer=Adam(learning_rate=0.0001),
    metrics=["mae"]
)

model_path = os.path.join(
    dataset_path,
    "Bestmodel.keras"
)

checkpoint = ModelCheckpoint(
    model_path,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=150,
    validation_data=(x_val, y_val),
    callbacks=[checkpoint],
    shuffle=True,
    verbose=1
)
