
# ============================================================
# CNN WHOLE-AREA LAND SUBSIDENCE PREDICTION
# ============================================================
#
# Input:
#   9 aligned raster layers
#
# CNN input:
#   30 x 30 x 9
#
# Output:
#   Whole-area subsidence prediction GeoTIFF
#
# The prediction raster keeps:
#   - same CRS
#   - same transform
#   - same resolution
#   - same dimensions
#   - same spatial extent
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import numpy as np
import rasterio as rio

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import load_model
from glob import glob

# ============================================================
# 2. PATHS
# ============================================================

root = os.path.dirname(os.getcwd())

dataset_path = os.path.join(
    root,
    "data"
)
dataset_path_raster = glob(os.path.join(dataset_path, '*_aligned.tif'))

model_path = os.path.join(
    dataset_path,
    "Bestmodel.keras"
)

output_path = os.path.join(
    'results',
    "CNN_subsidence_prediction.tif"
)


# ============================================================
# 3. PARAMETERS
# ============================================================

WINDOW_SIZE = 30

HALF_WINDOW = WINDOW_SIZE // 2

BATCH_SIZE = 4096

# ------------------------------------------------------------
# IMPORTANT
#
# During training:
# We have only subsidence (negative values), there aren't uplift in our data. So we used abs.
# If you have both uplift and subsidence, you don't use abs. You should use scaling that is appropriate for your data.
# ydata2 = abs(CUMUL_DISP / 10)
#
# Therefore model predictions are multiplied by -10
# to return to the original CUMUL_DISP units.
# ------------------------------------------------------------

TARGET_MULTIPLIER = -10.0


# ============================================================
# 4. INPUT RASTER ORDER
# ============================================================
#
# VERY IMPORTANT:
#
# These must be in EXACTLY the same order as the 9 channels
# used when train_data.npy was created.
# Do NOT mix original rasters such as:
#
#     aspect.tif
#
# with aligned rasters such as:
#
#     aspect_aligned.tif
#
# ============================================================
# Load raster data as driving forces of land subsidence
imgs = [rio.open(path) for path in dataset_path_raster]

print("Loaded raster datasets:")
for img in imgs:
    print(f" - {img.name} with shape {img.shape} and CRS {img.crs}")

# Extract image names from the dataset paths
raster_names = [os.path.basename(path) for path in dataset_path_raster]

# ============================================================
# 5. CHECK NUMBER OF INPUT RASTERS
# ============================================================

if len(raster_names) != 9:

    raise ValueError(
        f"Exactly 9 rasters are required. "
        f"Found {len(raster_names)}."
    )


# ============================================================
# 6. CHECK FILES EXIST
# ============================================================

raster_paths = []

for name in raster_names:

    path = os.path.join(
        dataset_path,
        name
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRaster not found:\n{path}"
        )

    raster_paths.append(path)


# ============================================================
# 7. LOAD TRAINED CNN MODEL
# ============================================================

print("\n============================================================")
print("Loading trained CNN model")
print("============================================================")

print("Model:", model_path)

model = load_model(
    model_path,
    compile=False
)

print("Model loaded successfully.")


# ============================================================
# 8. CHECK MODEL INPUT
# ============================================================

print("\nModel input shape:")
print(model.input_shape)

print("\nModel output shape:")
print(model.output_shape)


expected_shape = (None, 30, 30, 9)

if model.input_shape != expected_shape:

    print(
        "\nWARNING:"
        "\nExpected model input:"
        f" {expected_shape}"
        "\nActual model input:"
        f" {model.input_shape}"
    )


# ============================================================
# 9. OPEN RASTERS
# ============================================================

print("\n============================================================")
print("Opening input rasters")
print("============================================================")

rasters = {}

for name, path in zip(
    raster_names,
    raster_paths
):

    rasters[name] = rio.open(path)

    print(
        f"{name}: "
        f"{rasters[name].height} x "
        f"{rasters[name].width}"
    )


# ============================================================
# 10. DEFINE REFERENCE RASTER
# ============================================================
#
# The reference must be one of the ALIGNED rasters.
#
# Your preprocessing used the aligned water-table raster
# as the reference grid.
#
# ============================================================

reference_name = "watertable_aligned.tif"

reference = rasters[reference_name]


height = reference.height
width = reference.width

reference_transform = reference.transform
reference_crs = reference.crs


print("\n============================================================")
print("Reference raster")
print("============================================================")

print("Name:", reference_name)
print("Height:", height)
print("Width:", width)
print("CRS:", reference_crs)
print("Transform:")
print(reference_transform)


# ============================================================
# 11. CHECK ALL RASTERS ARE ALIGNED
# ============================================================

print("\n============================================================")
print("Checking raster alignment")
print("============================================================")

for name, raster in rasters.items():

    print(
        f"{name}: "
        f"shape={raster.shape}, "
        f"CRS={raster.crs}"
    )

    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    if raster.shape != reference.shape:

        raise ValueError(
            "\nShape mismatch:\n"
            f"{name}: {raster.shape}\n"
            f"Reference: {reference.shape}\n\n"
            "Use only the *_aligned.tif rasters."
        )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    if raster.transform != reference_transform:

        raise ValueError(
            "\nTransform mismatch:\n"
            f"{name}\n"
            f"Transform: {raster.transform}\n"
            f"Reference: {reference_transform}"
        )

    # --------------------------------------------------------
    # CRS
    # --------------------------------------------------------

    if raster.crs != reference_crs:

        raise ValueError(
            "\nCRS mismatch:\n"
            f"{name}: {raster.crs}\n"
            f"Reference: {reference_crs}"
        )


print("\nAll 9 rasters are perfectly aligned.")


# ============================================================
# 12. READ RASTERS
# ============================================================

print("\n============================================================")
print("Reading raster data")
print("============================================================")

raster_arrays = []

for name in raster_names:

    arr = rasters[name].read(
        1
    ).astype(
        np.float32
    )

    raster_arrays.append(
        arr
    )

    finite_values = arr[
        np.isfinite(arr)
    ]

    print(
        f"{name}: "
        f"shape={arr.shape}, "
        f"min={finite_values.min():.6f}, "
        f"max={finite_values.max():.6f}"
    )


# ============================================================
# 13. NORMALIZE RASTERS
# ============================================================
#
# This reproduces the normalization used in your training code:
#
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     normalized[name] = scaler.fit_transform(raster)
#
# ============================================================

print("\n============================================================")
print("Normalizing rasters")
print("============================================================")

normalized_arrays = []


for name, arr in zip(
    raster_names,
    raster_arrays
):

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    # --------------------------------------------------------
    # Match training preprocessing
    # --------------------------------------------------------

    normalized = scaler.fit_transform(
        arr
    ).astype(
        np.float32
    )

    normalized_arrays.append(
        normalized
    )

    print(
        f"{name}: "
        f"normalized min="
        f"{normalized.min():.6f}, "
        f"max="
        f"{normalized.max():.6f}"
    )


print("\nNormalization completed.")


# ============================================================
# 14. CREATE OUTPUT ARRAY
# ============================================================
#
# Initialize everything as NoData.
#
# Predictions will be written only where a complete
# 30 x 30 window is available.
#
# ============================================================

prediction_map = np.full(
    (height, width),
    np.nan,
    dtype=np.float32
)


# ============================================================
# 15. VALID PREDICTION AREA
# ============================================================

row_start = HALF_WINDOW

row_end = height - HALF_WINDOW

col_start = HALF_WINDOW

col_end = width - HALF_WINDOW


prediction_height = (
    row_end - row_start
)

prediction_width = (
    col_end - col_start
)

total_predictions = (
    prediction_height *
    prediction_width
)


print("\n============================================================")
print("Prediction area")
print("============================================================")

print(
    f"Rows:    {row_start} -> {row_end - 1}"
)

print(
    f"Columns: {col_start} -> {col_end - 1}"
)

print(
    f"Prediction size: "
    f"{prediction_height} x "
    f"{prediction_width}"
)

print(
    f"Total predictions: "
    f"{total_predictions:,}"
)


# ============================================================
# 16. PREPARE BATCH PREDICTION
# ============================================================
#
# Instead of calling model.predict() for every pixel,
# patches are extracted in large batches.
#
# This is much faster on the RTX 5080.
#
# ============================================================

print("\n============================================================")
print("Starting whole-area prediction")
print("============================================================")


# ------------------------------------------------------------
# Create row coordinates
# ------------------------------------------------------------

rows = np.arange(
    row_start,
    row_end,
    dtype=np.int32
)

cols = np.arange(
    col_start,
    col_end,
    dtype=np.int32
)


# ============================================================
# 17. PROCESS ROW BLOCKS
# ============================================================
#
# We process several rows at once.
#
# This avoids creating all ~9.4 million patches
# simultaneously in RAM.
#
# ============================================================

ROW_BLOCK_SIZE = 32

processed = 0


for block_start in range(
    0,
    len(rows),
    ROW_BLOCK_SIZE
):

    block_rows = rows[
        block_start:
        block_start + ROW_BLOCK_SIZE
    ]

    # --------------------------------------------------------
    # Number of rows in this block
    # --------------------------------------------------------

    n_rows = len(
        block_rows
    )

    n_cols = len(
        cols
    )

    n_samples = (
        n_rows *
        n_cols
    )

    # --------------------------------------------------------
    # Allocate block input
    #
    # Shape:
    #
    # (samples, 30, 30, 9)
    #
    # --------------------------------------------------------

    X_block = np.empty(
        (
            n_samples,
            WINDOW_SIZE,
            WINDOW_SIZE,
            9
        ),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Construct patches
    # --------------------------------------------------------

    sample_index = 0

    for row in block_rows:

        r0 = row - HALF_WINDOW

        r1 = row + HALF_WINDOW

        for col in cols:

            c0 = col - HALF_WINDOW

            c1 = col + HALF_WINDOW

            # ------------------------------------------------
            # Stack the nine raster channels
            # ------------------------------------------------

            for channel in range(9):

                X_block[
                    sample_index,
                    :,
                    :,
                    channel
                ] = normalized_arrays[channel][
                    r0:r1,
                    c0:c1
                ]

            sample_index += 1


    # ========================================================
    # 18. PREDICT CURRENT BLOCK IN BATCHES
    # ========================================================

    predictions = []

    for batch_start in range(
        0,
        n_samples,
        BATCH_SIZE
    ):

        batch_end = min(
            batch_start + BATCH_SIZE,
            n_samples
        )

        X_batch = X_block[
            batch_start:batch_end
        ]

        pred_batch = model.predict(
            X_batch,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        pred_batch = np.asarray(
            pred_batch
        ).reshape(-1)

        predictions.append(
            pred_batch
        )


    predictions = np.concatenate(
        predictions
    )


    # ========================================================
    # 19. CONVERT PREDICTIONS BACK TO ORIGINAL UNITS
    # ========================================================

    predictions *= TARGET_MULTIPLIER


    # ========================================================
    # 20. WRITE PREDICTIONS TO MAP
    # ========================================================

    prediction_index = 0

    for row in block_rows:

        for col in cols:

            prediction_map[
                row,
                col
            ] = predictions[
                prediction_index
            ]

            prediction_index += 1


    # ========================================================
    # 21. PROGRESS
    # ========================================================

    processed += n_samples

    percentage = (
        processed /
        total_predictions *
        100.0
    )

    print(
        f"Progress: "
        f"{percentage:6.2f}% "
        f"({processed:,} / "
        f"{total_predictions:,})"
    )


# ============================================================
# 22. PREDICTION STATISTICS
# ============================================================

valid_predictions = prediction_map[
    np.isfinite(prediction_map)
]


print("\n============================================================")
print("Prediction completed")
print("============================================================")

print(
    "Valid predictions:",
    len(valid_predictions)
)

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
# 23. CREATE OUTPUT PROFILE
# ============================================================

profile = reference.profile.copy()

profile.update(
    driver="GTiff",
    dtype="float32",
    count=1,
    height=height,
    width=width,
    crs=reference_crs,
    transform=reference_transform,
    nodata=-9999,
    compress="deflate",
    predictor=3
)


# ============================================================
# 24. CONVERT NaN TO NoData
# ============================================================

output_array = np.where(
    np.isfinite(prediction_map),
    prediction_map,
    -9999.0
).astype(
    np.float32
)


# ============================================================
# 25. SAVE GEOTIFF
# ============================================================

print("\n============================================================")
print("Saving prediction GeoTIFF")
print("============================================================")

with rio.open(
    output_path,
    "w",
    **profile
) as dst:

    dst.write(
        output_array,
        1
    )


# ============================================================
# 26. CLOSE INPUT RASTERS
# ============================================================

for raster in rasters.values():

    raster.close()


# ============================================================
# 27. FINAL MESSAGE
# ============================================================

print("\n============================================================")
print("DONE")
print("============================================================")

print(
    "Prediction map:"
)

print(
    output_path
)

print("\nRaster properties:")

print(
    "Dimensions:",
    height,
    "x",
    width
)

print(
    "CRS:",
    reference_crs
)

print(
    "NoData:",
    -9999
)

print(
    "Prediction range:",
    f"{np.min(valid_predictions):.4f}",
    "to",
    f"{np.max(valid_predictions):.4f}"
)

print("\nThe GeoTIFF is ready for QGIS / ArcGIS Pro.")

