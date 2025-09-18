GNILC: Generalized Needlet ILC for Component Separation

This repository provides a Python implementation of the GNILC algorithm for astrophysical component separation in multifrequency sky maps. It is particularly suitable for diffuse emission extraction in full-sky or large-area radio and microwave surveys.

## 📚 References

- Remazeilles et al. (2011), *MNRAS*, 410, 2481  
- Olivari, Remazeilles & Dickinson (2016), *MNRAS*, 456, 2749  
- Delabrouille et al. (2009), *A&A*, 493, 835

---

## Method Summary

The GNILC method performs:

1. **Needlet decomposition** of multifrequency sky maps.
2. **Local covariance analysis** with spatial masking.
3. **Whitening** of total covariance using prior (nuisance) components.
4. **AIC-based dimensionality selection**.
5. **ILC filtering** in the projected subspace.
6. **Wavelet synthesis** to produce the reconstructed GNILC maps.

---

## 📁 Input Files

Defined in `parameters.ini`, expected in `input/`:

- `input_maps.fits` — Observed maps [n_freq, n_pix]  
- `prior_maps.fits` — Prior component (e.g. synchrotron) [n_freq, n_pix]  
- `mask.fits` — Binary sky mask in HEALPix (RING)  
- `parameters.ini` — Main config file

Example `parameters.ini` entries:

[input]
input_maps = observed_cube
prior_maps = synchrotron_prior
mask = galactic_mask

[general]
needlet_bands = Cosine
bandcenters = 10,20,40,80,160
ilc_bias = 0.01
output_suffix = test01
save_ilc_weights = True
save_wavelets_dimensions = True



---

## 📤 Output Files

- `output/reconstructed_maps_<suffix>.fits` — GNILC-reconstructed maps  
- `output/wavelets_dimensions_<suffix>.fits` — (optional) AIC-derived dimension maps  
- Intermediate files (wavelet and weight FITS) are saved or deleted based on flags

---

## ⚙️ Dependencies

- Python ≥ 3.6  
- `numpy`  
- `healpy`  
- `astropy`  
- Custom modules: `gnilc_auxiliary.py`, `misc_functions.py`

---

## ▶️ How to Run

1. Edit `parameters.ini`
2. Place all `.fits` files in the `input/` directory
3. Run the main script:

python3 gnilc_main.py


Intermediate files are cleaned or saved depending on `parameters.ini` flags.

---

##  Notes

- Supports **Cosine** or **Gaussian** needlet windows  
- Assumes **RING-ordered** HEALPix maps  
- Prior maps must represent **nuisance** (e.g. synchrotron if extracting HI)  
- Designed for **large-area maps** (e.g. full-sky or wide MeerKAT fields)  
- Memory usage scales with `nside` and number of frequency channels

---

## 📄 License & Attribution

This implementation is provided for research use. If used in publications, please cite the relevant GNILC references listed above.
"""

