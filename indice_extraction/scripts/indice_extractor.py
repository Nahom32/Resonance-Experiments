import rasterio
import numpy as np
import json

path = "/Users/nahomsenay/Downloads/10_4231_R7RX991C/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R.tif"
dataset = rasterio.open(path)

# Read into memory [bands, height, width]
img = dataset.read()
print(img.shape)

meta = dataset.tags()
print(meta)


RED = img[34, :, :]  # adjust according to wavelength table
NIR = img[43, :, :]

ndvi = (NIR - RED) / (NIR + RED + 1e-6)
print(f"The ndvi is {ndvi}")


SWIR = img[100, :, :]  # example — adjust based on actual wavelengths
ndwi = (NIR - SWIR) / (NIR + SWIR)
print(f"The ndwi is {ndwi}")


def find_band(wavelengths, target):
    return int(np.argmin(np.abs(np.array(wavelengths) - target)))


meta = dataset.tags()  # read metadata dict
wavelengths = []
if "wavelength" in meta:
    wavelengths = json.loads(meta["wavelength"])
elif "WAVELENGTH" in meta:
    wavelengths = json.loads(meta["WAVELENGTH"])


idx_531 = find_band(wavelengths, 531)
idx_570 = find_band(wavelengths, 570)
idx_670 = find_band(wavelengths, 670)
idx_700 = find_band(wavelengths, 700)
idx_800 = find_band(wavelengths, 800)  # NIR
