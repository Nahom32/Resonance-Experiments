import rasterio
import numpy as np
import re
from pathlib import Path
from Py6S import SixS, AtmosProfile, AeroProfile, Geometry, Wavelength, AtmosCorr
# import matplotlib.pyplot as plt


def load_aviris_1992_wavelengths(calib_file: str):
    """
    Parses the 1992 AVIRIS 220-channel calibration text file and returns:
        wavelengths: np.array of shape (220,) with center wavelengths in nm
                     (NaN for the 5 bad/"not used" bands)
        good_bands: boolean mask (True = valid band)
        channel_to_band: mapping from data channel (0..219) → original AVIRIS band number
    """
    text = Path(calib_file).read_text()
    wavelengths = np.full(220, np.nan, dtype=float)
    channel_to_band = np.full(220, -1, dtype=int)
    pattern = re.compile(
        r"^\s*(\d+)\s+"  # group1: AVIRIS Band #
        r"(?:\((not used[^)]*)\)\s*|"  # group2: not used
        r"(\d+)\s+)"  # group3: Data Channel #
        r"([0-9.]+)\s+"  # group4: Center Wavelength (nm)
        r".*"  # ignore rest (FWHM, etc.)
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        band_str = match.group(1)
        band_num = int(band_str)
        if match.group(2):  # not used
            continue
        channel_str = match.group(3)
        wl_str = match.group(4)
        channel_idx = int(channel_str) - 1
        wavelengths[channel_idx] = float(wl_str)
        channel_to_band[channel_idx] = band_num
    good_bands = ~np.isnan(wavelengths)
    return wavelengths, good_bands, channel_to_band


# Step 1: Load hyperspectral TIFF (assume from previous stacking script)
base_path = "/Users/nahomsenay/Downloads/10_4231_R7RX991C/aviris_hyperspectral_data/"
input_path = base_path + "19920612_AVIRIS_IndianPine_EW-line_R.tif"
calib_file = "/Users/nahomsenay/Downloads/10_4231_R7RX991C/documentation/Calibration_Information_for_220_Channel_Data_Band_Set.txt"
wavelengths, good_bands, channel_to_band = load_aviris_1992_wavelengths(calib_file)
with rasterio.open(input_path, "r+") as dst:  # 'r+' mode to update metadata
    if dst.count != 220:
        raise ValueError("TIFF must have exactly 220 bands for AVIRIS 1992 data")
    dst.update_tags(wavelength=",".join(map(str, wavelengths)))
    print("Wavelength tag added successfully")
with rasterio.open(input_path) as src:
    data = src.read()  # (bands, height, width) - TOA radiance
    meta = src.meta.copy()
    wavelengths = [
        float(w) for w in src.tags()["wavelength"].split(",")
    ]  # nm to um later
# Step 2: Set up Py6S parameters (adjust based on metadata)
# Example: Mid-latitude summer, rural aerosols, sensor/view angles from metadata
# Step 2: Setup Py6S with atmospheric correction mode
s = SixS()
s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)
s.aero_profile = AeroProfile.PredefinedType(
    AeroProfile.Continental
)  # Valid type (replaces 'Rural')
s.aot550 = 0.2  # Adjust based on your scene
s.altitudes.set_sensor_satellite_level()  # Or custom for airborne
s.geometry = Geometry.User()
s.geometry.solar_z = 30  # From metadata
s.geometry.solar_a = 0
s.geometry.view_z = 0
s.geometry.view_a = 180

# Enable Lambertian atmospheric correction (assumes uniform surface; good for most ag scenes)
# Use a dummy radiance value (coefficients are independent of this input)
s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromRadiance(
    100.0
)  # Any positive value works

# Step 3: Per-band simulation to get coefficients
num_bands = data.shape[0]
xa = np.zeros(num_bands)
xb = np.zeros(num_bands)
xc = np.zeros(num_bands)

for i, wl_nm in enumerate(wavelengths):
    wl_um = wl_nm / 1000
    s.wavelength = Wavelength(wl_um)
    s.run()

    xa[i] = s.outputs.coef_xa
    xb[i] = s.outputs.coef_xb
    xc[i] = s.outputs.coef_xc

# Step 4: Apply correction using the formula
# rho = (xa * L_toa - xb) / (1 + xc * (xa * L_toa - xb))
corrected_data = np.zeros_like(data, dtype=np.float32)

for i in range(num_bands):
    L_toa = data[i]
    y = xa[i] * L_toa - xb[i]
    denom = 1 + xc[i] * y
    corrected_data[i] = np.divide(
        y, denom, where=denom != 0, out=np.full_like(y, np.nan)
    )

# Clip and handle invalids
corrected_data[corrected_data < 0] = 0
corrected_data[corrected_data > 1] = 1  # Reflectance bounded 0-1
corrected_data[np.isnan(corrected_data)] = np.nan

# Step 5: Save corrected TIFF
output_path = base_path + "atm_corrected.tif"
meta.update(dtype="float32")
with rasterio.open(output_path, "w", **meta) as dst:
    dst.write(corrected_data)
    dst.update_tags(wavelength=",".join(map(str, wavelengths)))  # Preserve wavelengths
    print(f"Corrected file saved to {output_path}")

# Step 6: Visualize example band or index (e.g., compute NDVI post-correction)
# ... (Use previous NDVI code on corrected_data)
