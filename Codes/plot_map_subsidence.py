# ============================================================
# CNN-Predicted Land Subsidence
# 6-Class Prediction Map
#
# Negative values = Subsidence
# Positive values = Uplift
#
# More negative -> greater subsidence
# More positive -> greater uplift
#
# Class 1 -> Very high subsidence -> Black
# Class 2 -> High subsidence      -> Dark red
# Class 3 -> Moderate subsidence  -> Red
# Class 4 -> Low subsidence       -> Yellow
# Class 5 -> Low uplift           -> Light blue
# Class 6 -> High uplift          -> Dark blue
# ============================================================

import os

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from rasterio.plot import plotting_extent


# ============================================================
# 1. Paths
# ============================================================

root = os.path.dirname(os.getcwd())

prediction_path = os.path.join(
    root,
    "results",
    "CNN_subsidence_prediction.tif"
)

output_path = os.path.join(
    root,
    "results",
    "subsidence_prediction_6_classes.png"
)


# ============================================================
# 2. Read prediction GeoTIFF
# ============================================================

with rasterio.open(prediction_path) as src:

    prediction = src.read(1)

    nodata = src.nodata

    extent = plotting_extent(src)

    crs = src.crs


# ============================================================
# 3. Handle NoData and invalid values
# ============================================================

if nodata is not None:

    prediction = np.where(
        prediction == nodata,
        np.nan,
        prediction
    )

prediction = np.where(
    np.isfinite(prediction),
    prediction,
    np.nan
)


# ============================================================
# 4. Get valid values
# ============================================================

valid = prediction[~np.isnan(prediction)]


# ============================================================
# 5. Statistics
# ============================================================

min_val = valid.min()
max_val = valid.max()
mean_val = valid.mean()
median_val = np.median(valid)

negative_values = valid[valid < 0]
positive_values = valid[valid > 0]

print()
print("Prediction statistics")
print("=====================")

print(f"Minimum : {min_val:.4f}")
print(f"Maximum : {max_val:.4f}")
print(f"Mean    : {mean_val:.4f}")
print(f"Median  : {median_val:.4f}")

print()
print("Number of values")
print("================")

print(f"Negative : {len(negative_values):,}")
print(f"Positive : {len(positive_values):,}")


# ============================================================
# 6. Define exactly 6 classes
#
# 4 classes = negative / subsidence
# 2 classes = positive / uplift
# ============================================================

if len(negative_values) == 0:

    raise ValueError(
        "No negative values were found in the prediction raster."
    )

if len(positive_values) == 0:

    raise ValueError(
        "No positive values were found in the prediction raster."
    )


# ------------------------------------------------------------
# Negative / subsidence boundaries
#
# 4 classes require 5 boundaries
# ------------------------------------------------------------

negative_breaks = np.linspace(
    negative_values.min(),
    0.0,
    5
)


# ------------------------------------------------------------
# Positive / uplift boundaries
#
# 2 classes require 3 boundaries
# ------------------------------------------------------------

positive_breaks = np.linspace(
    0.0,
    positive_values.max(),
    3
)


# ============================================================
# 7. Combine boundaries
#
# Negative boundaries:
#
# -580.63
# -435.47
# -290.31
# -145.15
#    0
#
# Positive boundaries:
#
#    0
#    1.80
#    3.61
#
# The duplicated zero is removed.
#
# Final:
#
# -580.63
# -435.47
# -290.31
# -145.15
#    0
#    1.80
#    3.61
#
# 7 boundaries = 6 classes
# ============================================================

class_breaks = np.concatenate([
    negative_breaks,
    positive_breaks[1:]
])


# ============================================================
# 8. Check boundaries
# ============================================================

n_classes = len(class_breaks) - 1

print()
print("Class boundaries")
print("================")

for i in range(len(class_breaks)):

    print(
        f"Boundary {i + 1}: "
        f"{class_breaks[i]:.4f}"
    )

print()
print(f"Number of boundaries: {len(class_breaks)}")
print(f"Number of classes:   {n_classes}")


if n_classes != 6:

    raise ValueError(
        f"The raster produced {n_classes} classes instead of 6."
    )


# ============================================================
# 9. Print class ranges
# ============================================================

print()
print("Six classes")
print("===========")

class_names = [
    "Very high subsidence",
    "High subsidence",
    "Moderate subsidence",
    "Low subsidence",
    "Low uplift",
    "High uplift"
]

for i in range(6):

    print(
        f"Class {i + 1}: "
        f"{class_names[i]:22s} "
        f"{class_breaks[i]:.2f} to "
        f"{class_breaks[i + 1]:.2f}"
    )


# ============================================================
# 10. Colour scheme
#
# Strong subsidence -> BLACK
#                   -> DARK RED
#                   -> RED
# Weak subsidence   -> YELLOW
#
# Uplift            -> LIGHT BLUE
# Strong uplift     -> DARK BLUE
# ============================================================

colors = [
    "#000000",   # Class 1 - Very high subsidence
    "#8B0000",   # Class 2 - High subsidence
    "#E34A33",   # Class 3 - Moderate subsidence
    "#FFD700",   # Class 4 - Low subsidence
    "#9ECAE1",   # Class 5 - Low uplift
    "#08519C"    # Class 6 - High uplift
]


# ============================================================
# 11. Create discrete colour map
# ============================================================

cmap = ListedColormap(colors)

norm = BoundaryNorm(
    class_breaks,
    ncolors=len(colors)
)


# ============================================================
# 12. Create figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 9)
)


# ============================================================
# 13. Plot prediction map
# ============================================================

ax.imshow(
    prediction,
    cmap=cmap,
    norm=norm,
    extent=extent,
    origin="upper"
)


# ============================================================
# 14. Title
# ============================================================

ax.set_title(
    "CNN-Predicted Land Subsidence",
    fontsize=18,
    fontweight="bold",
    pad=15
)


# ============================================================
# 15. Axis labels
# ============================================================

ax.set_xlabel(
    "Easting",
    fontsize=12
)

ax.set_ylabel(
    "Northing",
    fontsize=12
)


# ============================================================
# 16. Create legend
# ============================================================

legend_elements = []

for i in range(6):

    label = (
        f"{class_names[i]}: "
        f"{class_breaks[i]:.2f} – "
        f"{class_breaks[i + 1]:.2f}"
    )

    legend_elements.append(
        Patch(
            facecolor=colors[i],
            edgecolor="black",
            linewidth=0.8,
            label=label
        )
    )


# ============================================================
# 17. Add legend
# ============================================================

ax.legend(
    handles=legend_elements,
    title="Predicted Deformation",
    title_fontsize=12,
    fontsize=9.5,
    loc="lower right",
    frameon=True,
    framealpha=0.95,
    edgecolor="black"
)


# ============================================================
# 18. Grid
# ============================================================

ax.grid(
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)


# ============================================================
# 19. Layout
# ============================================================

plt.tight_layout()


# ============================================================
# 20. Save map
# ============================================================

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)


# ============================================================
# 21. Display
# ============================================================

plt.show()


# ============================================================
# 22. Output
# ============================================================

print()
print("Map saved successfully")
print("======================")
print(output_path)