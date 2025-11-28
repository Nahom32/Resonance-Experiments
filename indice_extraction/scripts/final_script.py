"""
====================================================
Extract Vegetation Indices from AVIRIS Hyperspectral
====================================================

This script:
✔ Reads AVIRIS hyperspectral TIFF
✔ Extracts wavelength metadata
✔ Maps wavelengths → band indices
✔ Computes vegetation indices (NDVI, PRI, MCARI)
✔ Saves results as GeoTIFF
====================================================
"""

import rasterio
import numpy as np
import json

# ----------------------------------------------
# CONFIG
# ----------------------------------------------
INPUT_FILE = "/Users/nahomsenay/Downloads/10_4231_R7RX991C/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R.tif"
OUTPUT_NDVI = "NDVI.tif"
OUTPUT_PRI = "PRI.tif"
OUTPUT_MCARI = "MCARI.tif"

# ----------------------------------------------
# Load AVIRIS Data
# ----------------------------------------------
dataset = rasterio.open(INPUT_FILE)
img = dataset.read()  # shape: (bands, H, W)
print(f"Loaded: {img.shape[0]} bands, size {img.shape[1]}x{img.shape[2]}")

# ----------------------------------------------
# Extract wavelength list
# ----------------------------------------------
meta = dataset.tags()
print(meta)
if "wavelength" in meta:
    wavelengths = json.loads(meta["wavelength"])
elif "WAVELENGTH" in meta:
    wavelengths = json.loads(meta["WAVELENGTH"])
elif dataset.descriptions:
    # try parsing descriptions
    wavelengths = []
    for d in dataset.descriptions:
        if d is not None and "nm" in d.lower():
            # expected format: "Wavelength=531.02 nm"
            w = "".join(c for c in d if c.isdigit() or c == ".")
            wavelengths.append(float(w))
else:
    raise Exception("❗ No wavelength metadata found. Check .hdr file.")

wavelengths = np.array(wavelengths)
print("Found wavelength list for all bands.")
print(wavelengths)


# ----------------------------------------------
# wavelength lookup helper
# ----------------------------------------------
def find_band(target):
    """Return index of band closest to a wavelength."""
    return int(np.argmin(np.abs(wavelengths - target)))


# ----------------------------------------------
# Assign band indices
# ----------------------------------------------
idx_RED = find_band(670)
idx_NIR = find_band(800)
idx_531 = find_band(531)
idx_570 = find_band(570)
idx_550 = find_band(550)
idx_700 = find_band(700)

print(f"RED band ≈ {wavelengths[idx_RED]} nm")
print(f"NIR band ≈ {wavelengths[idx_NIR]} nm")
print(f"531 band ≈ {wavelengths[idx_531]} nm")
print(f"570 band ≈ {wavelengths[idx_570]} nm")
print(f"550 band ≈ {wavelengths[idx_550]} nm")
print(f"700 band ≈ {wavelengths[idx_700]} nm")

# ----------------------------------------------
# Extract reflectance slices
# ----------------------------------------------
RED = img[idx_RED]
NIR = img[idx_NIR]
B531 = img[idx_531]
B570 = img[idx_570]
R550 = img[idx_550]
R700 = img[idx_700]

eps = 1e-6

# ----------------------------------------------
# Compute vegetation indices
# ----------------------------------------------

# NDVI
NDVI = (NIR - RED) / (NIR + RED + eps)

# PRI
PRI = (B531 - B570) / (B531 + B570 + eps)

# MCARI
MCARI = ((R700 - RED) - 0.2 * (R700 - R550)) / (R700 / (RED + eps) + eps)


# ----------------------------------------------
# Function to save output raster
# ----------------------------------------------
def save_raster(filename, arr):
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs=dataset.crs,
        transform=dataset.transform,
    ) as dst:
        dst.write(arr.astype("float32"), 1)
    print(f"Saved: {filename}")


# ----------------------------------------------
# Save results
# ----------------------------------------------
save_raster(OUTPUT_NDVI, NDVI)
save_raster(OUTPUT_PRI, PRI)
save_raster(OUTPUT_MCARI, MCARI)

print("✓ Completed vegetation index extraction successfully.")
