
# ============================================================
# Traditional Spatial Interpolation
# IDW + Local Ordinary Kriging + RBF
# ============================================================
#
# INPUT:
#   results/train_predictions.csv
#   results/test_predictions.csv
#
# REQUIRED COLUMNS:
#   Ground_truth
#   utmX
#   utmY
#
# OUTPUT:
#   results/train_predictions_interpolation.csv
#   results/test_predictions_interpolation.csv
#
# METHODS:
#   1. IDW
#   2. Local Ordinary Kriging
#   3. Local RBF
#
# IMPORTANT:
#
# The interpolation models are constructed ONLY from:
#
#   Training utmX
#   Training utmY
#   Training Ground_truth
#
# Test Ground_truth is NEVER used for interpolation.
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import time

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator

from pykrige.ok import OrdinaryKriging


# ============================================================
# 2. SETTINGS
# ============================================================

# ------------------------------------------------------------
# IDW
# ------------------------------------------------------------

IDW_POWER = 2.0

IDW_NEIGHBOURS = 12


# ------------------------------------------------------------
# Local Kriging
# ------------------------------------------------------------

KRIGING_NEIGHBOURS = 30

KRIGING_VARIogram = "spherical"


# ------------------------------------------------------------
# RBF
# ------------------------------------------------------------
#
# IMPORTANT:
#
# smoothing must NOT be zero for your dataset because duplicate
# or nearly duplicate coordinates can produce a singular matrix.
#
# ------------------------------------------------------------

RBF_NEIGHBOURS = 30

RBF_SMOOTHING = 0.01

RBF_KERNEL = "thin_plate_spline"


# ============================================================
# 3. PATHS
# ============================================================

root = os.path.dirname(os.getcwd())

result_path = os.path.join(
    root,
    "results"
)


train_file = os.path.join(
    result_path,
    "train_predictions.csv"
)


test_file = os.path.join(
    result_path,
    "test_predictions.csv"
)


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("LOADING DATA")
print("=" * 75)


if not os.path.exists(train_file):

    raise FileNotFoundError(
        f"\nTraining file not found:\n{train_file}"
    )


if not os.path.exists(test_file):

    raise FileNotFoundError(
        f"\nTesting file not found:\n{test_file}"
    )


train_dataset = pd.read_csv(
    train_file
)


test_dataset = pd.read_csv(
    test_file
)


print(
    f"\nTraining samples : "
    f"{len(train_dataset):,}"
)


print(
    f"Testing samples  : "
    f"{len(test_dataset):,}"
)


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Ground_truth",
    "utmX",
    "utmY"
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
    "\nRequired columns verified."
)


# ============================================================
# 6. EXTRACT ORIGINAL DATA
# ============================================================

train_x_all = train_dataset[
    "utmX"
].to_numpy(
    dtype=np.float64
)


train_y_all = train_dataset[
    "utmY"
].to_numpy(
    dtype=np.float64
)


train_values_all = train_dataset[
    "Ground_truth"
].to_numpy(
    dtype=np.float64
)


test_x_all = test_dataset[
    "utmX"
].to_numpy(
    dtype=np.float64
)


test_y_all = test_dataset[
    "utmY"
].to_numpy(
    dtype=np.float64
)


# ============================================================
# 7. VALID DATA MASKS
# ============================================================

train_valid = (
    np.isfinite(train_x_all)
    &
    np.isfinite(train_y_all)
    &
    np.isfinite(train_values_all)
)


test_valid = (
    np.isfinite(test_x_all)
    &
    np.isfinite(test_y_all)
)


print("\nData quality:")

print(
    f"Valid training points   : "
    f"{np.sum(train_valid):,}"
)


print(
    f"Invalid training points : "
    f"{np.sum(~train_valid):,}"
)


print(
    f"Valid testing points    : "
    f"{np.sum(test_valid):,}"
)


print(
    f"Invalid testing points  : "
    f"{np.sum(~test_valid):,}"
)


# ============================================================
# 8. VALID TRAINING DATA
# ============================================================

train_x = train_x_all[
    train_valid
]


train_y = train_y_all[
    train_valid
]


train_values = train_values_all[
    train_valid
]


train_coordinates = np.column_stack(
    (
        train_x,
        train_y
    )
)


# ============================================================
# 9. VALID TESTING DATA
# ============================================================

test_x = test_x_all[
    test_valid
]


test_y = test_y_all[
    test_valid
]


test_coordinates = np.column_stack(
    (
        test_x,
        test_y
    )
)


# ============================================================
# 10. REMOVE DUPLICATE TRAINING COORDINATES
# ============================================================
#
# This is particularly important for RBF and Kriging.
#
# If multiple observations have exactly the same UTM X/Y,
# they cannot all be treated as independent spatial locations.
#
# Here we aggregate duplicate coordinates using the mean
# Ground_truth value.
#
# ============================================================

print("\n" + "=" * 75)
print("CHECKING DUPLICATE TRAINING COORDINATES")
print("=" * 75)


training_dataframe = pd.DataFrame(
    {
        "utmX": train_x,
        "utmY": train_y,
        "Ground_truth": train_values
    }
)


number_before_duplicates = len(
    training_dataframe
)


training_dataframe = (
    training_dataframe
    .groupby(
        ["utmX", "utmY"],
        as_index=False
    )["Ground_truth"]
    .mean()
)


number_after_duplicates = len(
    training_dataframe
)


duplicates_removed = (
    number_before_duplicates
    -
    number_after_duplicates
)


print(
    f"Training observations before : "
    f"{number_before_duplicates:,}"
)


print(
    f"Unique spatial locations     : "
    f"{number_after_duplicates:,}"
)


print(
    f"Duplicate observations removed: "
    f"{duplicates_removed:,}"
)


# ------------------------------------------------------------
# Final interpolation source
# ------------------------------------------------------------

source_x = training_dataframe[
    "utmX"
].to_numpy(
    dtype=np.float64
)


source_y = training_dataframe[
    "utmY"
].to_numpy(
    dtype=np.float64
)


source_values = training_dataframe[
    "Ground_truth"
].to_numpy(
    dtype=np.float64
)


source_coordinates = np.column_stack(
    (
        source_x,
        source_y
    )
)


print(
    f"\nUnique interpolation points: "
    f"{len(source_coordinates):,}"
)


# ============================================================
# 11. IDW FUNCTION
# ============================================================

def idw_predict(
    source_coordinates,
    source_values,
    target_coordinates,
    power=2.0,
    k=12
):
    """
    Inverse Distance Weighting interpolation.
    """

    source_coordinates = np.asarray(
        source_coordinates,
        dtype=np.float64
    )


    source_values = np.asarray(
        source_values,
        dtype=np.float64
    )


    target_coordinates = np.asarray(
        target_coordinates,
        dtype=np.float64
    )


    tree = cKDTree(
        source_coordinates
    )


    k = min(
        k,
        len(source_coordinates)
    )


    distances, indices = tree.query(
        target_coordinates,
        k=k
    )


    if k == 1:

        distances = distances[
            :,
            None
        ]

        indices = indices[
            :,
            None
        ]


    predictions = np.empty(
        len(target_coordinates),
        dtype=np.float64
    )


    for i in range(
        len(target_coordinates)
    ):

        d = distances[i]

        idx = indices[i]


        # ----------------------------------------------------
        # Exact spatial match
        # ----------------------------------------------------

        zero_distance = (
            d == 0
        )


        if np.any(zero_distance):

            predictions[i] = (
                source_values[
                    idx[
                        np.where(
                            zero_distance
                        )[0][0]
                    ]
                ]
            )

            continue


        # ----------------------------------------------------
        # IDW weights
        # ----------------------------------------------------

        weights = 1.0 / (
            d ** power
        )


        predictions[i] = (
            np.sum(
                weights *
                source_values[idx]
            )
            /
            np.sum(weights)
        )


    return predictions


# ============================================================
# 12. IDW INTERPOLATION
# ============================================================

print("\n" + "=" * 75)
print("IDW INTERPOLATION")
print("=" * 75)


print(
    f"Power      : {IDW_POWER}"
)


print(
    f"Neighbours : {IDW_NEIGHBOURS}"
)


start_time = time.time()


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

print(
    "\nCalculating IDW training predictions..."
)


train_IDW_valid = idw_predict(
    source_coordinates,
    source_values,
    train_coordinates,
    power=IDW_POWER,
    k=IDW_NEIGHBOURS
)


# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------

print(
    "Calculating IDW testing predictions..."
)


test_IDW_valid = idw_predict(
    source_coordinates,
    source_values,
    test_coordinates,
    power=IDW_POWER,
    k=IDW_NEIGHBOURS
)


print(
    f"\nIDW completed in "
    f"{time.time() - start_time:.2f} seconds."
)


# ============================================================
# 13. LOCAL ORDINARY KRIGING FUNCTION
# ============================================================

def local_ordinary_kriging(
    source_coordinates,
    source_values,
    target_coordinates,
    neighbours=30,
    variogram_model="spherical"
):
    """
    Local Ordinary Kriging.

    Only the nearest training observations are used for each
    target point.

    This avoids the huge computational cost of global
    Ordinary Kriging.
    """

    source_coordinates = np.asarray(
        source_coordinates,
        dtype=np.float64
    )


    source_values = np.asarray(
        source_values,
        dtype=np.float64
    )


    target_coordinates = np.asarray(
        target_coordinates,
        dtype=np.float64
    )


    tree = cKDTree(
        source_coordinates
    )


    k = min(
        neighbours,
        len(source_coordinates)
    )


    distances, indices = tree.query(
        target_coordinates,
        k=k
    )


    if k == 1:

        distances = distances[
            :,
            None
        ]

        indices = indices[
            :,
            None
        ]


    predictions = np.full(
        len(target_coordinates),
        np.nan,
        dtype=np.float64
    )


    total = len(
        target_coordinates
    )


    for i in range(total):

        if (
            i % 500 == 0
            or
            i == total - 1
        ):

            percentage = (
                100.0
                *
                (i + 1)
                /
                total
            )


            print(
                f"Kriging: "
                f"{i + 1:,}/{total:,} "
                f"({percentage:.1f}%)"
            )


        local_indices = indices[i]


        local_x = source_coordinates[
            local_indices,
            0
        ]


        local_y = source_coordinates[
            local_indices,
            1
        ]


        local_values = source_values[
            local_indices
        ]


        # ----------------------------------------------------
        # Remove duplicate local coordinates
        # ----------------------------------------------------

        local_coordinates = np.column_stack(
            (
                local_x,
                local_y
            )
        )


        _, unique_indices = np.unique(
            local_coordinates,
            axis=0,
            return_index=True
        )


        local_x = local_x[
            unique_indices
        ]


        local_y = local_y[
            unique_indices
        ]


        local_values = local_values[
            unique_indices
        ]


        # ----------------------------------------------------
        # Need at least 3 points
        # ----------------------------------------------------

        if len(local_values) < 3:

            predictions[i] = np.mean(
                local_values
            )

            continue


        target_x = (
            target_coordinates[i, 0]
        )


        target_y = (
            target_coordinates[i, 1]
        )


        # ----------------------------------------------------
        # Exact coordinate
        # ----------------------------------------------------

        exact_distance = np.sqrt(
            (
                local_x -
                target_x
            ) ** 2
            +
            (
                local_y -
                target_y
            ) ** 2
        )


        if np.any(
            exact_distance < 1e-10
        ):

            predictions[i] = (
                local_values[
                    np.argmin(
                        exact_distance
                    )
                ]
            )

            continue


        # ----------------------------------------------------
        # Ordinary Kriging
        # ----------------------------------------------------

        try:

            OK = OrdinaryKriging(
                local_x,
                local_y,
                local_values,

                variogram_model=variogram_model,

                verbose=False,

                enable_plotting=False,

                coordinates_type="euclidean"
            )


            prediction, variance = (
                OK.execute(
                    "points",
                    np.array(
                        [target_x]
                    ),
                    np.array(
                        [target_y]
                    )
                )
            )


            prediction_value = float(
                prediction[0]
            )


            if np.isfinite(
                prediction_value
            ):

                predictions[i] = (
                    prediction_value
                )

            else:

                predictions[i] = np.mean(
                    local_values
                )


        except Exception:

            # ------------------------------------------------
            # Fallback to local mean
            # ------------------------------------------------

            predictions[i] = np.mean(
                local_values
            )


    return predictions


# ============================================================
# 14. LOCAL ORDINARY KRIGING
# ============================================================

print("\n" + "=" * 75)
print("LOCAL ORDINARY KRIGING")
print("=" * 75)


print(
    f"Neighbours : "
    f"{KRIGING_NEIGHBOURS}"
)


print(
    f"Variogram  : "
    f"{KRIGING_VARIogram}"
)


start_time = time.time()


# ------------------------------------------------------------
# Training Kriging
# ------------------------------------------------------------

print(
    "\nCalculating Kriging training predictions..."
)


train_Kriging_valid = local_ordinary_kriging(
    source_coordinates,
    source_values,
    train_coordinates,
    neighbours=KRIGING_NEIGHBOURS,
    variogram_model=KRIGING_VARIogram
)


# ------------------------------------------------------------
# Testing Kriging
# ------------------------------------------------------------

print(
    "\nCalculating Kriging testing predictions..."
)


test_Kriging_valid = local_ordinary_kriging(
    source_coordinates,
    source_values,
    test_coordinates,
    neighbours=KRIGING_NEIGHBOURS,
    variogram_model=KRIGING_VARIogram
)


print(
    f"\nKriging completed in "
    f"{time.time() - start_time:.2f} seconds."
)


# ============================================================
# 15. RBF INTERPOLATION
# ============================================================

print("\n" + "=" * 75)
print("RADIAL BASIS FUNCTION INTERPOLATION")
print("=" * 75)


print(
    f"Kernel     : {RBF_KERNEL}"
)


print(
    f"Neighbours : {RBF_NEIGHBOURS}"
)


print(
    f"Smoothing  : {RBF_SMOOTHING}"
)


start_time = time.time()


# ------------------------------------------------------------
# Build RBF model
# ------------------------------------------------------------

print(
    "\nBuilding RBF model..."
)


rbf_model = RBFInterpolator(
    source_coordinates,
    source_values,

    kernel=RBF_KERNEL,

    neighbors=RBF_NEIGHBOURS,

    smoothing=RBF_SMOOTHING,

    degree=1
)


print(
    "RBF model built successfully."
)


# ------------------------------------------------------------
# RBF training predictions
# ------------------------------------------------------------

print(
    "\nCalculating RBF training predictions..."
)


try:

    train_RBF_valid = rbf_model(
        train_coordinates
    )


except np.linalg.LinAlgError:

    print(
        "\nWARNING: RBF encountered a singular "
        "matrix during training prediction."
    )


    print(
        "Retrying with stronger smoothing..."
    )


    rbf_model = RBFInterpolator(
        source_coordinates,
        source_values,

        kernel=RBF_KERNEL,

        neighbors=RBF_NEIGHBOURS,

        smoothing=0.1,

        degree=1
    )


    train_RBF_valid = rbf_model(
        train_coordinates
    )


print(
    "Training RBF completed."
)


# ------------------------------------------------------------
# RBF testing predictions
# ------------------------------------------------------------

print(
    "\nCalculating RBF testing predictions..."
)


try:

    test_RBF_valid = rbf_model(
        test_coordinates
    )


except np.linalg.LinAlgError:

    print(
        "\nWARNING: RBF encountered a singular "
        "matrix during testing prediction."
    )


    print(
        "Retrying with stronger smoothing..."
    )


    rbf_model = RBFInterpolator(
        source_coordinates,
        source_values,

        kernel=RBF_KERNEL,

        neighbors=RBF_NEIGHBOURS,

        smoothing=0.1,

        degree=1
    )


    test_RBF_valid = rbf_model(
        test_coordinates
    )


print(
    "Testing RBF completed."
)


print(
    f"\nRBF completed in "
    f"{time.time() - start_time:.2f} seconds."
)


# ============================================================
# 16. CONVERT RBF OUTPUTS
# ============================================================

train_RBF_valid = np.asarray(
    train_RBF_valid,
    dtype=np.float64
)


test_RBF_valid = np.asarray(
    test_RBF_valid,
    dtype=np.float64
)


train_IDW_valid = np.asarray(
    train_IDW_valid,
    dtype=np.float64
)


test_IDW_valid = np.asarray(
    test_IDW_valid,
    dtype=np.float64
)


train_Kriging_valid = np.asarray(
    train_Kriging_valid,
    dtype=np.float64
)


test_Kriging_valid = np.asarray(
    test_Kriging_valid,
    dtype=np.float64
)


# ============================================================
# 17. CREATE FULL-SIZE OUTPUT ARRAYS
# ============================================================

train_IDW_prediction = np.full(
    len(train_dataset),
    np.nan,
    dtype=np.float64
)


train_Kriging_prediction = np.full(
    len(train_dataset),
    np.nan,
    dtype=np.float64
)


train_RBF_prediction = np.full(
    len(train_dataset),
    np.nan,
    dtype=np.float64
)


test_IDW_prediction = np.full(
    len(test_dataset),
    np.nan,
    dtype=np.float64
)


test_Kriging_prediction = np.full(
    len(test_dataset),
    np.nan,
    dtype=np.float64
)


test_RBF_prediction = np.full(
    len(test_dataset),
    np.nan,
    dtype=np.float64
)


# ============================================================
# 18. INSERT PREDICTIONS
# ============================================================

train_IDW_prediction[
    train_valid
] = train_IDW_valid


train_Kriging_prediction[
    train_valid
] = train_Kriging_valid


train_RBF_prediction[
    train_valid
] = train_RBF_valid


test_IDW_prediction[
    test_valid
] = test_IDW_valid


test_Kriging_prediction[
    test_valid
] = test_Kriging_valid


test_RBF_prediction[
    test_valid
] = test_RBF_valid


# ============================================================
# 19. ADD PREDICTIONS TO DATASETS
# ============================================================

train_dataset[
    "IDW_Prediction"
] = train_IDW_prediction


train_dataset[
    "Kriging_Prediction"
] = train_Kriging_prediction


train_dataset[
    "RBF_Prediction"
] = train_RBF_prediction


test_dataset[
    "IDW_Prediction"
] = test_IDW_prediction


test_dataset[
    "Kriging_Prediction"
] = test_Kriging_prediction


test_dataset[
    "RBF_Prediction"
] = test_RBF_prediction


# ============================================================
# 20. CHECK PREDICTIONS
# ============================================================

print("\n" + "=" * 75)
print("PREDICTION QUALITY CHECK")
print("=" * 75)


print("\nTRAINING:")


print(
    f"IDW     valid predictions : "
    f"{np.sum(np.isfinite(train_IDW_prediction)):,}"
)


print(
    f"Kriging valid predictions : "
    f"{np.sum(np.isfinite(train_Kriging_prediction)):,}"
)


print(
    f"RBF     valid predictions : "
    f"{np.sum(np.isfinite(train_RBF_prediction)):,}"
)


print("\nTESTING:")


print(
    f"IDW     valid predictions : "
    f"{np.sum(np.isfinite(test_IDW_prediction)):,}"
)


print(
    f"Kriging valid predictions : "
    f"{np.sum(np.isfinite(test_Kriging_prediction)):,}"
)


print(
    f"RBF     valid predictions : "
    f"{np.sum(np.isfinite(test_RBF_prediction)):,}"
)


# ------------------------------------------------------------
# Prediction ranges
# ------------------------------------------------------------

print("\nPrediction ranges:")


print(
    f"\nIDW:"
)


print(
    f"  Train: "
    f"{np.nanmin(train_IDW_prediction):.6f}"
    f" to "
    f"{np.nanmax(train_IDW_prediction):.6f}"
)


print(
    f"  Test : "
    f"{np.nanmin(test_IDW_prediction):.6f}"
    f" to "
    f"{np.nanmax(test_IDW_prediction):.6f}"
)


print(
    f"\nKriging:"
)


print(
    f"  Train: "
    f"{np.nanmin(train_Kriging_prediction):.6f}"
    f" to "
    f"{np.nanmax(train_Kriging_prediction):.6f}"
)


print(
    f"  Test : "
    f"{np.nanmin(test_Kriging_prediction):.6f}"
    f" to "
    f"{np.nanmax(test_Kriging_prediction):.6f}"
)


print(
    f"\nRBF:"
)


print(
    f"  Train: "
    f"{np.nanmin(train_RBF_prediction):.6f}"
    f" to "
    f"{np.nanmax(train_RBF_prediction):.6f}"
)


print(
    f"  Test : "
    f"{np.nanmin(test_RBF_prediction):.6f}"
    f" to "
    f"{np.nanmax(test_RBF_prediction):.6f}"
)


# ============================================================
# 21. SAVE OUTPUT FILES
# ============================================================

train_output = os.path.join(
    result_path,
    "train_predictions_interpolation.csv"
)


test_output = os.path.join(
    result_path,
    "test_predictions_interpolation.csv"
)


print("\n" + "=" * 75)
print("SAVING RESULTS")
print("=" * 75)


train_dataset.to_csv(
    train_output,
    index=False
)


print(
    f"\nTraining results saved to:\n"
    f"{train_output}"
)


test_dataset.to_csv(
    test_output,
    index=False
)


print(
    f"\nTesting results saved to:\n"
    f"{test_output}"
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("INTERPOLATION COMPLETED")
print("=" * 75)


print(
    "\nMethods:"
)


print(
    "  1. IDW"
)


print(
    "  2. Local Ordinary Kriging"
)


print(
    "  3. Local RBF"
)


print(
    "\nTraining predictions:"
)


print(
    f"  {len(train_IDW_valid):,} IDW"
)


print(
    f"  {len(train_Kriging_valid):,} Kriging"
)


print(
    f"  {len(train_RBF_valid):,} RBF"
)


print(
    "\nTesting predictions:"
)


print(
    f"  {len(test_IDW_valid):,} IDW"
)


print(
    f"  {len(test_Kriging_valid):,} Kriging"
)


print(
    f"  {len(test_RBF_valid):,} RBF"
)


print(
    "\nOutput:"
)


print(
    train_output
)


print(
    test_output
)


print(
    "\nDone."
)
