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
import re
from pathlib import Path
import os
import matplotlib.pyplot as plt  # ----------------------------------------------

# CONFIG
# ----------------------------------------------
INPUT_FILE = "/Users/nahomsenay/Downloads/10_4231_R7RX991C/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R.tif"
OUTPUT_NDVI = "NDVI.tif"
OUTPUT_PRI = "PRI.tif"
OUTPUT_MCARI = "MCARI.tif"
METADATA_FILE = "/Users/nahomsenay/Downloads/10_4231_R7RX991C/documentation/Calibration_Information_for_220_Channel_Data_Band_Set.txt"


def load_aviris_1992_wavelengths(calib_file: str = METADATA_FILE):
    """
    Parses the 1992 AVIRIS 220-channel calibration text file and returns:
        wavelengths: np.array of shape (220,) with center wavelengths in nm
                     (NaN for the 5 bad/"not used" bands)
        good_bands: boolean mask (True = valid band)
        channel_to_band: mapping from data channel (0..219) → original AVIRIS band number
    """
    text = Path(calib_file).read_text()

    wavelengths = np.full(220, np.nan, dtype=float)

    # Regex to catch lines like:
    #   14      13     517.98    10.05       1.00            0.50
    pattern = re.compile(
        r"^\s*\d+\s+"  # AVIRIS Band # (ignored)
        r"(?:\((not used[^)]*)\)\s+|"  # either "(not used ...)"  OR
        r"(\d+)\s+)"  # Data Channel #
        r"([0-9.]+)\s+"  # Center Wavelength (nm)
    )

    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue

        # If it's a "not used" line
        if match.group(1):  # "(not used ..."
            continue

        channel_str = match.group(2)
        wl_str = match.group(3)

        channel_idx = int(channel_str) - 1  # data channel 1..220 → 0..219
        wavelength = float(wl_str)

        wavelengths[channel_idx] = wavelength

    good_bands = ~np.isnan(wavelengths)

    return wavelengths, good_bands


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
print(f"The meta file is: {meta}")
# if "wavelength" in meta:
#    wavelengths = json.loads(meta["wavelength"])
# elif "WAVELENGTH" in meta:
#    wavelengths = json.loads(meta["WAVELENGTH"])
# elif dataset.descriptions:
#    # try parsing descriptions
#    wavelengths = []
#    for d in dataset.descriptions:
#        if d is not None and "nm" in d.lower():
#            # expected format: "Wavelength=531.02 nm"
#            w = "".join(c for c in d if c.isdigit() or c == ".")
#            wavelengths.append(float(w))
# else:
#    raise Exception("❗ No wavelength metadata found. Check .hdr file.")
#
# wavelengths = np.array(wavelengths)
# print("Found wavelength list for all bands.")
# print(wavelengths)

wavelengths, good_mask = load_aviris_1992_wavelengths()


# ----------------------------------------------
# wavelength lookup helper
# ----------------------------------------------
def find_band(target_nm: float, name: str = "") -> int:
    """Returns the closest valid band index (0..219)"""
    diffs = np.abs(wavelengths - target_nm)
    idx = int(np.argmin(diffs))

    if np.isnan(wavelengths[idx]):
        raise ValueError(f"No valid band near {target_nm} nm ({name})")

    return idx


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
b = {
    "blue": 4,  # 429.43 nm
    "green": 17,  # 557.49 nm   ← best green peak
    "red": 29,  # 666.61 nm
    "rededge": 38,  # 725.47 nm   ← sweet spot for RE indices
    "nir": 45,  # 802.53 nm   ← classic NIR plateau
    "swir1": 126,  # 1581.30 nm  ← for NDWI/NDMI (1600 nm type)
    "swir2": 192,  # 2270.15 nm  ← for deeper water absorption
}
# NDVI
NDVI = (NIR - RED) / (NIR + RED + eps)

# PRI
PRI = (B531 - B570) / (B531 + B570 + eps)

# MCARI
MCARI = ((R700 - RED) - 0.2 * (R700 - R550)) / (R700 / (RED + eps) + eps)

# Extract once (assuming `img` is your (rows, cols, 220) array)
B = img[:, :, b["blue"] - 1]
G = img[:, :, b["green"] - 1]
R = img[:, :, b["red"] - 1]
RE = img[:, :, b["rededge"] - 1]
NIR = img[:, :, b["nir"] - 1]
SWIR1 = img[:, :, b["swir1"] - 1]
SWIR2 = img[:, :, b["swir2"] - 1]

gndvi = (NIR - G) / (NIR + G + 1e-8)
ndre = (NIR - RE) / (NIR + RE + 1e-8)
mtci = (NIR - RE) / (RE - R + 1e-8)
cire = NIR / RE - 1
cig = NIR / G - 1
evi = 2.5 * (NIR - R) / (NIR + 6 * R - 7.5 * B + 1 + 1e-8)
savi = 1.5 * (NIR - R) / (NIR + R + 0.5 + 1e-8)
osavi = (NIR - R) / (NIR + R + 0.16 + 1e-8)
ndwi_1600 = (NIR - SWIR1) / (NIR + SWIR1 + 1e-8)
ndwi_2200 = (NIR - SWIR2) / (NIR + SWIR2 + 1e-8)
vari = (G - R) / (G + R - B + 1e-8)
tgi = -0.5 * (190 * (R - G) - 120 * (R - B))
ccci = ndre / ((NIR - R) / (NIR + R + 1e-8))  # NDRE / NDVI
cvi = (NIR * R) / (G**2 + 1e-6)


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
# save_raster(OUTPUT_NDVI, NDVI)
# save_raster(OUTPUT_PRI, PRI)
# save_raster(OUTPUT_MCARI, MCARI)
# os.makedirs("indices", exist_ok=True)

# Your dictionary of already-computed indices (2D numpy arrays)
indices = {
    "GNDVI": gndvi,
    "NDVI": NDVI,
    "NDRE": ndre,
    "MTCI": mtci,
    "CIre": cire,
    "CIg": cig,
    "EVI": evi,
    "SAVI": savi,
    "OSAVI": osavi,
    "NDWI_1600": ndwi_1600,
    "NDWI_2200": ndwi_2200,
    "VARI": vari,
    "TGI": tgi,
    "CCCI": ccci,
    "CVI": cvi,
}

colormaps = {
    "NDVI": "RdYlGn",
    "GNDVI": "RdYlGn",
    "NDRE": "plasma",
    "MTCI": "viridis",
    "CIre": "magma",
    "CIg": "cividis",
    "EVI": "RdYlGn",
    "SAVI": "RdYlGn",
    "OSAVI": "RdYlGn",
    "NDWI_1600": "Blues_r",
    "NDWI_2200": "Blues_r",
    "VARI": "RdBu",
    "TGI": "RdYlGn_r",
    "CCCI": "turbo",
}

os.makedirs("indices_pretty", exist_ok=True)

for name, array in indices.items():
    clean = np.nan_to_num(array, nan=np.nan)  # keep actual NaN, not -9999 for plotting

    plt.figure(figsize=(10, 8))
    cmap = colormaps.get(name, "viridis")  # fallback
    vmin = np.nanpercentile(clean, 2)
    vmax = np.nanpercentile(clean, 98)

    plt.imshow(clean, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(name, fontsize=20)
    plt.axis("off")
    plt.colorbar(shrink=0.7, label=name)

    out_png = f"indices_pretty/{name}.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close()

    # Also save a beautiful stretched GeoTIFF (8-bit) for QGIS
    stretched = np.interp(clean, (vmin, vmax), (1, 255)).astype("uint8")
    filename = f"indices_pretty/{name}_8bit.tif"
    save_raster(filename, stretched)
print("✓ Completed vegetation index extraction successfully.")
