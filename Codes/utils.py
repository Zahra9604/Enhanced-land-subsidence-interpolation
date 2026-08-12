
import numpy as np
from rasterio.warp import reproject, Resampling
import rasterio as rio
import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial import cKDTree

def align_raster(src_raster, ref_transform, ref_crs, ref_shape, ref_profile, output_raster):
    """Align src_raster to match the reference raster's grid, CRS, and shape."""
    
    with rio.open(src_raster) as src:
        src_data = src.read(1)

        # Prepare empty destination array with reference shape
        dst_data = np.empty(ref_shape, dtype=src_data.dtype)

        # Reproject to reference grid
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest,
        )

    # Update profile to match reference raster
    aligned_profile = ref_profile.copy()
    aligned_profile.update(
        height=ref_shape[0],
        width=ref_shape[1],
        dtype="float32",
        count=1,
        nodata=np.nan
    )

    # Write aligned raster
    with rio.open(output_raster, 'w', **aligned_profile) as dst:
        dst.write(dst_data, 1)



# Trim the raster to match the required shape
def trim_raster(src_raster, output_raster):
    """Trim the raster to match the required shape."""
    with rio.open(src_raster) as src:
        # Read the data from the raster
        data = src.read(1)
        # Trim the last row if the raster has one more row
        data_trimmed = data[:-1, :]

        # Copy the profile and adjust height
        profile = src.profile
        profile.update(height=data_trimmed.shape[0])

        # Save the trimmed raster
        with rio.open(output_raster, 'w', **profile) as dst:
            dst.write(data_trimmed, 1)


def save_predictions(
    file_path,
    utmx,
    utmy,
    predictions,
    ground_truth
):

    with open(
        file_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "utmX",
            "utmY",
            "Prediction",
            "Ground_truth"
        ])

        if utmx is None or utmy is None:

            for prediction, truth in zip(
                predictions,
                ground_truth
            ):

                writer.writerow([
                    prediction,
                    truth
                ])

        else:

            for x, y, prediction, truth in zip(
                utmx,
                utmy,
                predictions,
                ground_truth
            ):

                writer.writerow([
                    x,
                    y,
                    prediction,
                    truth
                ])

def load_coordinates(split,dataset_path):

    x_path = os.path.join(
        dataset_path,
        f"utmx_{split}.npy"
    )

    y_path = os.path.join(
        dataset_path,
        f"utmy_{split}.npy"
    )

    if not (
        os.path.exists(x_path)
        and os.path.exists(y_path)
    ):
        return None, None

    return (
        np.load(x_path),
        np.load(y_path)
    )


def calculate_metrics(y_true, y_pred):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mse, rmse, mae, r2


def idw_predict(
    source_coordinates,
    source_values,
    target_coordinates,
    power=2.0,
    k=12
):
    """
    Inverse Distance Weighting (IDW).

    Parameters
    ----------
    source_coordinates : ndarray
        Coordinates of known observations, shape (n, 2).

    source_values : ndarray
        Known values at source coordinates.

    target_coordinates : ndarray
        Coordinates where predictions are required.

    power : float
        IDW distance power.

    k : int
        Number of nearest neighbours.

    Returns
    -------
    predictions : ndarray
        IDW predictions.
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

    # KD-tree for efficient nearest-neighbour search
    tree = cKDTree(source_coordinates)

    k = min(k, len(source_coordinates))

    distances, indices = tree.query(
        target_coordinates,
        k=k
    )

    # When k=1, make arrays 2-dimensional
    if k == 1:

        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    predictions = np.empty(
        len(target_coordinates),
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Handle exact coordinate matches
    # --------------------------------------------------------

    exact_match = np.any(
        distances == 0,
        axis=1
    )

    # Normal IDW prediction for non-exact points
    non_exact = ~exact_match

    if np.any(non_exact):

        d = distances[non_exact]
        idx = indices[non_exact]

        weights = 1.0 / np.power(
            d,
            power
        )

        predictions[non_exact] = np.sum(
            weights * source_values[idx],
            axis=1
        ) / np.sum(
            weights,
            axis=1
        )

    # Exact coordinate -> use observed value
    if np.any(exact_match):

        exact_rows = np.where(exact_match)[0]

        for row in exact_rows:

            zero_index = np.where(
                distances[row] == 0
            )[0][0]

            predictions[row] = source_values[
                indices[row, zero_index]
            ]

    return predictions

