# ============================================================
# Required libraries
# ============================================================

import os
import numpy as np
import rasterio as rio
from rasterio.windows import Window
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from glob import glob

# ============================================================
# Paths
# ============================================================

root = os.path.dirname(os.getcwd())
dataset_path = os.path.join(root, "data")

model_path = os.path.join(
    dataset_path,
    "Bestmodel.keras"
)

output_path = os.path.join(
    root,'results',
    "subsidence_prediction.tif"
)


# ============================================================
# Parameters
# ============================================================

WINDOW_SIZE = 30
HALF_WINDOW = WINDOW_SIZE // 2

BATCH_SIZE = 128

# Your training labels were:
# ydata2 = CUMUL_DISP / 10
#
# Therefore, multiply predictions by 10 if you
# want the original CUMUL_DISP units.
OUTPUT_MULTIPLIER = -10.0


# ============================================================
# Load trained model
# ============================================================

print("Loading trained model...")

model = load_model(model_path)

print("Model loaded successfully.")


# ============================================================
# Load the 9 aligned rasters
# ============================================================

# IMPORTANT:
# These must be the SAME 9 rasters and in the SAME ORDER
# as used when creating train_data.npy.
# ============================================================
# Extract image names from the dataset paths
# ============================================================
# Load raster data as drivinf forces of land subsidence
dataset_path_raster = glob(os.path.join(dataset_path, '*_aligned.tif'))
imgs = [rio.open(path) for path in dataset_path_raster]

print("Loaded raster datasets:")
for img in imgs:
    print(f" - {img.name} with shape {img.shape} and CRS {img.crs}")

image_names = [
    os.path.basename(path)
    for path in dataset_path_raster
]

# Create dictionary:
# filename → raster dataset

rasters = dict(zip(image_names, imgs))


# ============================================================
# Check raster alignment
# ============================================================

# Use the first raster as the reference
reference_name = image_names[0]
reference = rasters[reference_name]

height = reference.height
width = reference.width

reference_transform = reference.transform
reference_crs = reference.crs

print("\nReference raster:")
print("Name:", reference_name)
print("Height:", height)
print("Width:", width)
print("CRS:", reference_crs)
print("Transform:", reference_transform)


# ============================================================
# Check all rasters against reference
# ============================================================

for name, raster in rasters.items():

    if raster.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch:\n"
            f"{raster.name}: {raster.shape}\n"
            f"Reference: {reference.shape}"
        )

    if raster.transform != reference_transform:
        raise ValueError(
            f"Transform mismatch:\n"
            f"{raster.name}"
        )

    if raster.crs != reference_crs:
        raise ValueError(
            f"CRS mismatch:\n"
            f"{raster.name}"
        )


print("\nAll rasters are aligned.")


# ============================================================
# Read raster data
# ============================================================

print("\nReading rasters...")

raster_arrays = []

for name, raster in rasters.items():

    arr = raster.read(1).astype(np.float32)

    raster_arrays.append(arr)

    print(
        f"{name}: "
        f"shape={arr.shape}, "
        f"min={np.nanmin(arr):.4f}, "
        f"max={np.nanmax(arr):.4f}"
    )
# ============================================================
# Check number of input rasters
# ============================================================

if len(raster_paths) != 9:
    raise ValueError(
        f"Expected 9 input rasters, but found {len(raster_paths)}."
    )


# ============================================================
# Open rasters
# ============================================================

rasters = []

for path in raster_paths:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raster not found:\n{path}"
        )

    rasters.append(rio.open(path))


print("\nInput rasters:")

for raster in rasters:
    print(
        f"{os.path.basename(raster.name)} | "
        f"shape={raster.shape} | "
        f"CRS={raster.crs}"
    )


# ============================================================
# Check raster alignment
# ============================================================

reference = rasters[0]

height = reference.height
width = reference.width

reference_transform = reference.transform
reference_crs = reference.crs

print("\nReference raster:")
print("Height:", height)
print("Width:", width)
print("CRS:", reference_crs)
print("Transform:", reference_transform)


for raster in rasters:

    if raster.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch:\n"
            f"{raster.name}: {raster.shape}\n"
            f"Reference: {reference.shape}"
        )

    if raster.transform != reference_transform:
        raise ValueError(
            f"Transform mismatch:\n"
            f"{raster.name}"
        )

    if raster.crs != reference_crs:
        raise ValueError(
            f"CRS mismatch:\n"
            f"{raster.name}"
        )


print("\nAll rasters are aligned.")


# ============================================================
# Read raster data
# ============================================================

print("\nReading rasters...")

raster_arrays = []

for raster in rasters:

    arr = raster.read(1).astype(np.float32)

    raster_arrays.append(arr)

    print(
        f"{os.path.basename(raster.name)}: "
        f"min={np.nanmin(arr):.4f}, "
        f"max={np.nanmax(arr):.4f}"
    )


# ============================================================
# Normalize rasters
# ============================================================

print("\nNormalizing rasters...")

normalized_arrays = []

for arr in raster_arrays:

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    # IMPORTANT:
    # This reproduces your training code:
    #
    # scaler.fit_transform(raster)

    normalized = scaler.fit_transform(arr)

    normalized = normalized.astype(
        np.float32
    )

    normalized_arrays.append(normalized)


print("Normalization completed.")


# ============================================================
# Create prediction map
# ============================================================

prediction_map = np.full(
    (height, width),
    np.nan,
    dtype=np.float32
)


# ============================================================
# Valid prediction area
# ============================================================

# Because the model requires a complete 30 × 30 window,
# predictions cannot be made for the outer 15 pixels.

row_start = HALF_WINDOW
row_end = height - HALF_WINDOW

col_start = HALF_WINDOW
col_end = width - HALF_WINDOW


print("\nPrediction area:")
print(
    f"Rows: {row_start} → {row_end - 1}"
)

print(
    f"Columns: {col_start} → {col_end - 1}"
)


# ============================================================
# Generate predictions row by row
# ============================================================

print("\nStarting whole-area prediction...")

for row in range(row_start, row_end):

    batch_x = []
    batch_locations = []

    for col in range(col_start, col_end):

        r0 = row - HALF_WINDOW
        r1 = row + HALF_WINDOW

        c0 = col - HALF_WINDOW
        c1 = col + HALF_WINDOW

        # ----------------------------------------------------
        # Extract 30 × 30 × 9 patch
        # ----------------------------------------------------

        patch = np.stack(
            [
                normalized_arrays[channel][
                    r0:r1,
                    c0:c1
                ]
                for channel in range(9)
            ],
            axis=-1
        )

        batch_x.append(patch)
        batch_locations.append((row, col))

        # ----------------------------------------------------
        # Predict when batch is full
        # ----------------------------------------------------

        if len(batch_x) == BATCH_SIZE:

            batch_x = np.asarray(
                batch_x,
                dtype=np.float32
            )

            predictions = model.predict(
                batch_x,
                verbose=0
            ).reshape(-1)

            # Convert back to original target units
            predictions *= OUTPUT_MULTIPLIER

            for (r, c), prediction in zip(
                batch_locations,
                predictions
            ):
                prediction_map[r, c] = prediction

            batch_x = []
            batch_locations = []

    # --------------------------------------------------------
    # Predict remaining patches in this row
    # --------------------------------------------------------

    if len(batch_x) > 0:

        batch_x = np.asarray(
            batch_x,
            dtype=np.float32
        )

        predictions = model.predict(
            batch_x,
            verbose=0
        ).reshape(-1)

        predictions *= OUTPUT_MULTIPLIER

        for (r, c), prediction in zip(
            batch_locations,
            predictions
        ):
            prediction_map[r, c] = prediction

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if row % 25 == 0:

        progress = (
            (row - row_start)
            / (row_end - row_start)
            * 100
        )

        print(
            f"Progress: {progress:.1f}% "
            f"({row}/{row_end - 1})"
        )


# ============================================================
# Prediction statistics
# ============================================================

valid_predictions = prediction_map[
    ~np.isnan(prediction_map)
]

print("\nPrediction completed.")

print(
    "Minimum:",
    np.min(valid_predictions)
)

print(
    "Maximum:",
    np.max(valid_predictions)
)

print(
    "Mean:",
    np.mean(valid_predictions)
)

print(
    "Median:",
    np.median(valid_predictions)
)


# ============================================================
# Save prediction as GeoTIFF
# ============================================================

print("\nSaving prediction map...")

profile = reference.profile.copy()

profile.update(
    dtype="float32",
    count=1,
    compress="deflate",
    predictor=3,
    nodata=-9999
)


# Replace NaN with NoData value
output_array = np.where(
    np.isnan(prediction_map),
    -9999,
    prediction_map
).astype(np.float32)


with rio.open(
    output_path,
    "w",
    **profile
) as dst:

    dst.write(
        output_array,
        1
    )


print("\n============================================================")
print("Prediction map saved successfully!")
print("============================================================")
print(output_path)