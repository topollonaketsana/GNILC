
# Foreground Cube 

This script generates simulated full-sky foreground maps across a range of frequencies in HEALPix format, suitable for intensity mapping and component separation studies in radio and microwave astronomy.

## Description

Each foreground component (e.g., synchrotron, free-free, AME, thermal dust, point sources, CMB) can be independently configured, simulated, and optionally saved as a 3D frequency cube.

The script uses models defined in a configuration file (`parameters.ini`) and generates outputs in a local `output/` directory.

---

## Configuration

All parameters are defined in `parameters.ini`. Sections include:

- `[General]`: Frequency range, nside, and output label  
- `[Synchrotron]`, `[FreeFree]`, `[AME]`, `[ThermalDust]`, `[PointSources]`, `[CMB]`: Component-specific parameters

### Example `parameters.ini`

```ini
[General]
freq_min = 100
freq_width = 5
nchannels = 100
nside = 128
output_suffix = test

[Synchrotron]
simulate = True
model_template = synch_map.fits
spectral_index_model = beta_map.fits
curvature_index = -0.05
curvature_reference_freq = 408
save_cube = True

[CMB]
simulate = True
cmb_model = commander_template.fits
save_cube = True
```

---

## Components Supported

- **Synchrotron**
- **Free-Free**
- **Anomalous Microwave Emission (AME)**
- **Thermal Dust**
- **Unresolved Point Sources**
- **CMB**

Each component adds to the total foreground cube. Individual cubes are optionally saved.

---

## Output

- `output/foreground_cube_<suffix>.fits` — Total emission cube  
- `output/<component>_cube_<suffix>.fits` — Individual component cubes (if enabled)  

All outputs are 3D arrays with shape `(n_channels, n_pixels)` in HEALPix RING ordering.

---

## Running the Script

```bash
python3 generate_foregrounds.py
```

Make sure `parameters.ini` and all required input maps (e.g., templates, spectral index maps) are present in the working directory.

---

## Dependencies

- **Python** ≥ 3.6  
- **numpy**  
- **healpy**  
- **astropy**  

**Custom modules:**

- `foregrounds_functions.py`  
- `misc_functions.py`

---

## Use Case

Designed for simulation-based studies of component separation techniques (e.g., GNILC, PCA) in experiments like MeerKAT, SKA, or CMB satellite missions.

---

## License & Attribution

This code is for research purposes. If used in a publication, please acknowledge the simulation framework or cite relevant component models.
