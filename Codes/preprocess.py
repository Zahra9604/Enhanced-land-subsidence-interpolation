
# Required libraries

from email.mime import image
from utils import align_raster, trim_raster
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import pandas as pd
import  pyproj
import rasterio as rio
import numpy as np
from sklearn.model_selection import train_test_split
import csv
import tensorflow as tf
from tensorflow import keras as ks
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from keras.regularizers import l2 , l1
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Activation, MaxPooling2D, Dropout, AveragePooling2D, GlobalMaxPooling2D
# import livelossplot
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import cv2
from rasterio.warp import reproject, Resampling
from glob import glob
from rasterio.warp import reproject, Resampling

# load PS data
Dir = os.getcwd()
root = os.path.dirname(Dir)
dataset_path = os.path.join(root,'data')
Ps_data_path = os.path.join(dataset_path,'PSnoup.xls')
df = pd.read_excel(Ps_data_path)

# Define the cumulative displacement as the target variable
ydata = df['CUMUL_DISP']
# Define the UTM projection for zone 39N (WGS84)
p = pyproj.Proj(proj='utm', zone=39, ellps='WGS84')
# Convert latitude and longitude to UTM coordinates
lat=df['LAT']
lon=df['LON']
lat=list(lat)
lon=list(lon)
utmx, utmy = p(lon,lat)
# Load raster data as driving forces of land subsidence
dataset_path_raster = glob(os.path.join(dataset_path, '*.tif'))
imgs = [rio.open(path) for path in dataset_path_raster]

print("Loaded raster datasets:")
for img in imgs:
    print(f" - {img.name} with shape {img.shape} and CRS {img.crs}")

# Extract image names from the dataset paths
image_names = [os.path.basename(path) for path in dataset_path_raster]
# Create a dictionary to map image names to their corresponding raster datasets
image_dict = dict(zip(image_names, imgs))
# Read band 1 from each raster and store in a dictionary
band1_dict = {name: img.read(1) for name, img in image_dict.items()}
# We can now access the raster data using the image_dict and band1_dict dictionaries. For example, to access the band-1 array of "NDVI.tif", we can use band1_dict["NDVI.tif"].
# print("Loaded band-1 arrays:")
# for name, arr in band1_dict.items():
#     print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")

# We have to ensure that all rasters have the same shape and alignment. If they don't, we may need to resample or reproject them to match a reference raster.
#  Trim water table raster to match the shape of other rasters


# Create a trimmed version of the water table raster

trim_raster(image_dict["watertable.tif"].name, os.path.join(dataset_path, "watertable_aligned.tif"))
print("Trimmed water table raster saved as 'watertable_aligned.tif'.")

# We should check the transform, CRS, and shape of all rasters to ensure they are aligned. If they are not aligned, we may need to resample or reproject them to match a reference raster.
# Pick a reference raster (e.g., the first one)


watertable_path = os.path.join(dataset_path, "watertable_aligned.tif")
ref_raster = rio.open(watertable_path)
wt_shape = (ref_raster.height, ref_raster.width)
wt_bounds = ref_raster.bounds
wt_crs = ref_raster.crs
wt_transform = ref_raster.transform
wt_profile = ref_raster.profile

    
# print("Reference raster (watertable_trimmed.tif) properties:")
# print("Watertable shape:", wt_shape)
# print("Watertable bounds:", wt_bounds)
# print("Watertable CRS:", wt_crs)
# print("Watertable transform:", wt_transform)
# print("Watertable profile:", wt_profile)




image_names_2 = image_names[:-1]  # Exclude the reference raster from the list of image names
image_dict_2 = {name: image_dict[name] for name in image_names_2}  # Exclude the reference raster from the dictionary
# Align rasters to the reference raster's grid, CRS, and shape

# ============================================================
# Water-table_trimmed raster = REFERENCE GRID with 3043*3164 dimensions
# ============================================================
for name, raster in image_dict_2.items():
    src_path = raster.name
    output_path = os.path.join(dataset_path, name.replace(".tif", "_aligned.tif"))

    align_raster(
        src_path,
        wt_transform,
        wt_crs,
        wt_shape,
        wt_profile,
        output_path
    )

    print(f"Aligned {name} → {output_path}")
# ============================================================
#  Load aligned rasters
# ============================================================

aligned_dict = {}
image_names_aligned = [name.replace(".tif", "_aligned.tif") for name in image_names_2]
for name in image_names_aligned:

    aligned_path = os.path.join(dataset_path,name)

    aligned_dict[name] = rio.open(aligned_path)

# Add the trimmed water-table reference raster
aligned_dict["watertable_aligned.tif"] = ref_raster
image_names_aligned = list(aligned_dict.keys())
# ============================================================
#  Load band-1 arrays
# ============================================================

aligned_band1 = {
    name: ds.read(1)
    for name, ds in aligned_dict.items()
}


# # ============================================================
#  Check aligned rasters
# # ============================================================

# print("Aligned rasters loaded:")

# for name, ds in aligned_dict.items():
#     print(
#         f"{name}: "
#         f"shape={ds.shape}, "
#         f"CRS={ds.crs}"
#     )


# ============================================================
#  Check that all rasters have the same grid
# ============================================================

reference = aligned_dict["NDVI_aligned.tif"]

for name, ds in aligned_dict.items():

    shape_ok = ds.shape == reference.shape
    transform_ok = ds.transform == reference.transform
    crs_ok = ds.crs == reference.crs

    print(
        f"{name}: "
        f"Shape={shape_ok}, "
        f"Transform={transform_ok}, "
        f"CRS={crs_ok}"
    )


# # ============================================================
#  Optional: test coordinate → pixel conversion
# # ============================================================

# x_coord = 582569.6114867731
# y_coord = 3547085.9505179366

# row, col = reference.index(x_coord, y_coord)

# print(
#     f"\nTest coordinate: ({x_coord}, {y_coord})"
# )
# print(
#     f"Reference pixel: Row={row}, Column={col}"
# )
# rw,cw = aligned_dict [ 'watertable_aligned.tif'].index(x_coord, y_coord)
# print(f"Water-table pixel: Row={rw}, Column={cw}")

# ============================================================
#  Find PS pixel locations
# ============================================================

WINDOW_SIZE = 30
HALF_WINDOW = WINDOW_SIZE // 2

ps_pixels = []

for i, (x, y) in enumerate(zip(utmx, utmy)):

    row, col = reference.index(x, y)

    # Keep only PS points where the complete
    # 30 × 30 window fits inside the raster
    if (
        HALF_WINDOW <= row < reference.height - HALF_WINDOW
        and
        HALF_WINDOW <= col < reference.width - HALF_WINDOW
    ):
        ps_pixels.append((row, col, i))


# ============================================================
#  Get valid PS indices
# ============================================================

valid_indices = np.array(
    [ps_index for _, _, ps_index in ps_pixels],
    dtype=np.int32
)

print("\nPS point statistics:")
print(f"Total PS points:   {len(utmx)}")
print(f"Valid PS points:   {len(valid_indices)}")
print(f"Removed PS points: {len(utmx) - len(valid_indices)}")


# ============================================================
#  Prepare target variable and coordinates
# ============================================================
# Our data has just subsidence values (negative values)
ydata2 = (
    abs(np.asarray(ydata)[valid_indices])
    .astype(np.float32)
    / 10.0
)
#UTM coordinates for valid PS points
utmx2 = np.asarray(utmx)[valid_indices].astype(np.float32) 
utmy2 = np.asarray(utmy)[valid_indices].astype(np.float32)


# ============================================================
#  Allocate 30 × 30 feature arrays
# ============================================================

n_samples = len(ps_pixels)

data = {
    name: np.empty(
        (n_samples, WINDOW_SIZE, WINDOW_SIZE),
        dtype=np.float32
    )
    for name in aligned_band1
}


# ============================================================
#  Extract 30 × 30 windows
# ============================================================

for sample_idx, (row, col, _) in enumerate(ps_pixels):

    row_start = row - HALF_WINDOW
    row_end = row + HALF_WINDOW

    col_start = col - HALF_WINDOW
    col_end = col + HALF_WINDOW

    for name, raster_array in aligned_band1.items():

        data[name][sample_idx] = raster_array[
            row_start:row_end,
            col_start:col_end
        ]


# ============================================================
#  Check extracted data
# ============================================================

print("\nExtracted feature shapes:")

for name, array in data.items():
    print(f"{name}: {array.shape}")

print("\nTarget and coordinate shapes:")
print(f"ydata2: {ydata2.shape}")
print(f"utmx2:  {utmx2.shape}")
print(f"utmy2:  {utmy2.shape}")

# ============================================================
# Normalize aligned rasters
# ============================================================

normalized = {}
scalers = {}

for name in image_names_aligned:

    raster = aligned_band1[name]

    scaler = MinMaxScaler(feature_range=(0, 1))

    # MinMaxScaler expects 2D input
    normalized[name] = scaler.fit_transform(
        raster
    ).astype(np.float32)

    scalers[name] = scaler

# ============================================================
# Check normalized rasters
# ============================================================

print("Normalized raster ranges:")

for name in image_names_aligned:

    print(
        f"min={normalized[name].min():.4f}, "
        f"max={normalized[name].max():.4f}"
    )

# ============================================================
# # Create CNN input tensor
# # Shape: [samples, channels, rows, columns]
# ============================================================

Data = np.empty(
                (
                len(ps_pixels),
                len(image_names_aligned),
                WINDOW_SIZE,
                WINDOW_SIZE
                ),
                dtype=np.float32
                )

for sample_idx, (row, col, _) in enumerate(ps_pixels):

    r0 = row - HALF_WINDOW
    r1 = row + HALF_WINDOW
    c0 = col - HALF_WINDOW
    c1 = col + HALF_WINDOW

    for channel, name in enumerate(image_names_aligned):

        Data[sample_idx, channel] = normalized[name][
            r0:r1,
            c0:c1
        ]


print("Final input shape:", Data.shape)

indices = np.arange(len(ps_pixels))
#split  test data
x_train, x_test, y_train, y_test ,indices_train, indices_test = train_test_split(Data, ydata2,indices, test_size=0.1, shuffle=True)
#split validation data
trainshape=x_train.shape
indices2 = np.arange(trainshape[0])
x_train2, x_val, y_train2, y_val  ,indices_train2, indices_val = train_test_split(x_train, y_train, indices2, test_size=0.1, shuffle=True)
# transposing
train_data = np.transpose(x_train2, (0, 2, 3, 1))
test_data   = np.transpose(x_test,   (0, 2, 3, 1))
val_data    = np.transpose(x_val,    (0, 2, 3, 1))

# Save the datasets for later use
np.save(os.path.join(dataset_path, 'train_data.npy'), train_data)
np.save(os.path.join(dataset_path, 'test_data.npy'), test_data)
np.save(os.path.join(dataset_path, 'val_data.npy'), val_data)

np.save(os.path.join(dataset_path, 'train_label.npy'), y_train2)
np.save(os.path.join(dataset_path, 'test_label.npy'), y_test)
np.save(os.path.join(dataset_path, 'val_label.npy'), y_val)

np.save(os.path.join(dataset_path, 'ind_train.npy'), indices_train2)
np.save(os.path.join(dataset_path, 'ind_test.npy'), indices_test)
np.save(os.path.join(dataset_path, 'ind_val.npy'), indices_val)

# Create lists for utmx and utmy for training data
utmx_train = [utmx2[i] for i in indices_train2]
utmy_train = [utmy2[i] for i in indices_train2]
y_train_real = [ydata2[i] for i in indices_train2]
# Create lists for utmx and utmy for testing data
utmx_test = [utmx2[i] for i in indices_test]
utmy_test = [utmy2[i] for i in indices_test]
y_test_real = [ydata2[i] for i in indices_test]
# Create lists for utmx and utmy for validation data
utmx_val = [utmx2[i] for i in indices_val]
utmy_val = [utmy2[i] for i in indices_val]
y_val_real = [ydata2[i] for i in indices_val]
# Save the UTM coordinates and real target values for each split to use in evaluation 
np.save(os.path.join(dataset_path, 'utmx_train.npy'), utmx_train)
np.save(os.path.join(dataset_path, 'utmy_train.npy'), utmy_train)
np.save(os.path.join(dataset_path, 'utmx_test.npy'), utmx_test)
np.save(os.path.join(dataset_path, 'utmy_test.npy'), utmy_test)
np.save(os.path.join(dataset_path, 'utmx_val.npy'), utmx_val)
np.save(os.path.join(dataset_path, 'utmy_val.npy'), utmy_val)
np.save(os.path.join(dataset_path, 'train_label_real.npy'), y_train_real)
np.save(os.path.join(dataset_path, 'test_label_real.npy'), y_test_real)
np.save(os.path.join(dataset_path, 'val_label_real.npy'), y_val_real)