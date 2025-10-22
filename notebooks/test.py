# make_inputs.py  — GNILC input preparation
import numpy as np
import healpy as hp
import os
import matplotlib.pyplot as plt
from astropy.io import fits

# -------------------------------
# CONFIG
# -------------------------------
N_SELECT = 5  # number of frequency channels to keep
path_21cm = "../21 cm/map_n-2_nc-252_ns-256_bw-mwa-32float_nreal-1.fits"
path_synch = "../Synch/DGSE_n-2_nc-252_ns-256_bw-mwa-32float_syncmod-1.fits"
output_dir = "../data/"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# LOAD MAPS
# -------------------------------
with fits.open(path_21cm) as hdul:
    map_21cm = hdul[0].data.astype(np.float32)

with fits.open(path_synch) as hdul:
    map_synch = hdul[0].data.astype(np.float32)

print("Loaded:")
print(f" 21 cm map shape: {map_21cm.shape}")
print(f" Synch map shape: {map_synch.shape}")

# -------------------------------
# CONVERT SYNCH MAP TO mK
# -------------------------------
# (If synch map is in Kelvin → multiply by 1e3)
map_synch_mK = map_synch * 1e3
print("✅ Converted synchrotron map from K to mK")

# -------------------------------
# REDUCE NUMBER OF CHANNELS
# -------------------------------
n_ch_total = map_21cm.shape[0]
indices = np.linspace(0, n_ch_total - 1, N_SELECT, dtype=int)

map_21cm_red = map_21cm[indices, :]
map_synch_red = map_synch_mK[indices, :]

print(f"Reduced to {N_SELECT} channels:", indices)

# -------------------------------
# DEFINE FREQUENCIES
# -------------------------------580–1015 MHz
freq_min = 120.0  # MHz, lowest frequency
freq_max = 180.0  # MHz, highest frequency

freqs = np.linspace(freq_min, freq_max, N_SELECT)
print("Frequencies for selected channels (MHz):", freqs)

# Save frequencies to a FITS file
freqs_fname = os.path.join(output_dir, "freqs.fits")
fits.PrimaryHDU(freqs).writeto(freqs_fname, overwrite=True)
print("✅ Frequencies saved to", freqs_fname)

# -------------------------------
# CHECK BASIC STATS
# -------------------------------
print('21 cm min/max (mK):', np.min(map_21cm_red), np.max(map_21cm_red))
print('Synch min/max (mK):', np.min(map_synch_red), np.max(map_synch_red))

# -------------------------------
# COMBINE MAPS
# -------------------------------
total_map = map_21cm_red + map_synch_red
prior = map_21cm_red.copy()
nside = 256  # from your original maps
npix = hp.nside2npix(nside)

print("Total map shape:", total_map.shape)

# -------------------------------
# SAVE GNILC-READY 2D FITS
# -------------------------------
prior_fname_gnilc = os.path.join(output_dir, "prior_21cm_gnilc.fits")
total_fname_gnilc = os.path.join(output_dir, "total_map_gnilc.fits")

fits.PrimaryHDU(prior).writeto(prior_fname_gnilc, overwrite=True)
fits.PrimaryHDU(total_map).writeto(total_fname_gnilc, overwrite=True)
print("✅ Saved GNILC-ready 2D FITS maps")

# -------------------------------
# SAVE HEALPY-FORMAT MAPS (for plotting)
# -------------------------------
prior_fname_hp = os.path.join(output_dir, "prior_21cm_hp.fits")
total_fname_hp = os.path.join(output_dir, "total_map_hp.fits")

hp.write_map(prior_fname_hp, prior, dtype=np.float32, overwrite=True)
hp.write_map(total_fname_hp, total_map, dtype=np.float32, overwrite=True)
print("✅ Saved Healpy-format maps for plotting")

# -------------------------------
# FULL-SKY MASK
# -------------------------------
mask = np.ones(npix, dtype=np.float32)
mask_fname = os.path.join(output_dir, "mask.fits")
hp.write_map(mask_fname, mask, overwrite=True)
print("✅ Full-sky mask saved")

# -------------------------------
# SUMMARY
# -------------------------------
print("\nSummary:")
print(f"Channels kept: {N_SELECT}")
print(f"NSIDE: {nside}, NPIX: {npix}")
print(f"Prior mean (mK): {np.mean(prior):.3e}")
print(f"Total map mean (mK): {np.mean(total_map):.3e}")

# -------------------------------
# PLOT CHANNEL 0 and SAVE PNG
# -------------------------------
hp.mollview(hp.ud_grade(hp.read_map(prior_fname_hp, field=0), 128),
            title="Prior 21 cm (ch0)", unit="mK")
plt.savefig(os.path.join(output_dir, "prior_ch0.png"))
plt.close()

hp.mollview(hp.ud_grade(hp.read_map(total_fname_hp, field=0), 128),
            title="Total (21 cm + synch) (ch0)", unit="mK")
plt.savefig(os.path.join(output_dir, "total_ch0.png"))
plt.close()

print("✅ Plots saved as PNG in", output_dir)
