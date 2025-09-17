# Simulated Foregrounds & GNILC Component Separation

This repository provides two core tools for radio and microwave sky analysis:

- **Foreground Cube Generator**: simulates diffuse astrophysical foregrounds across frequency.  
- **GNILC Processor**: performs component separation using the Generalized Needlet ILC method.

Together, these tools enable realistic sky simulations and advanced component extraction techniques suitable for intensity mapping and CMB experiments.

---

##  Contents

### 1. Foreground Cube Generator

Simulates full-sky foreground emission in HEALPix format. Supports:

- Synchrotron emission with curvature
- Free-free emission
- AME (Anomalous Microwave Emission)
- Thermal dust
- Point sources (with optional clustering)
- CMB

Each component is configurable via `parameters.ini`.

Script: `generate_foregrounds.py`  
Output: `output/foreground_cube_<suffix>.fits`, individual component cubes (optional)

---

### 2. GNILC Component Separation

Performs component separation on multifrequency maps using the Generalized Needlet ILC method.

Main features:

- Needlet-based multiscale filtering
- Local PCA and AIC dimensionality estimation
- Prior-informed subspace projection
- Full map reconstruction

Script: `gnilc_main.py`  
Output: `output/reconstructed_maps_<suffix>.fits`

Requires input maps (`input_maps.fits`, `prior_maps.fits`) and a sky mask.

---

## How to Use

1. **Configure `parameters.ini`** for foreground simulation
2. Run foreground generator:

```bash
python3 generate_foregrounds.py
