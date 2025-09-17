# Rewritten Foreground Cube Generator
# Author: Adapted by ChatGPT (OpenAI)

import numpy as np
import healpy as hp
import configparser
from astropy.io import fits as pyfits
import os

import foregrounds_functions as fg
import misc_functions as utils

# Load configuration
cfg = configparser.ConfigParser()
param_file = "parameters.ini"
section = "General"

# Parse experiment setup
params = utils.ConfigSectionMap(cfg, param_file, section)

freq_min = float(params['freq_min'])
freq_width = float(params['freq_width'])
nchannels = int(params['nchannels'])
nside = int(params['nside'])
output_suffix = params['output_suffix']

frequencies = freq_min + np.arange(nchannels) * freq_width

# Setup output directory
output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output')
os.makedirs(output_dir, exist_ok=True)

# Initialize total cube
npix = hp.nside2npix(nside)
total = np.zeros((nchannels, npix), dtype=np.float32)

# Helper function to write cube

def write_cube(label, cube):
    filename = f"{output_dir}/{label}_cube_{output_suffix}.fits"
    pyfits.writeto(filename, cube, overwrite=True)

# Synchrotron
if utils.ConfigGetBoolean(cfg, param_file, "Synchrotron", 'simulate'):
    s_cfg = utils.ConfigSectionMap(cfg, param_file, "Synchrotron")
    cube = fg.synchrotron(
        frequencies, nside,
        template_model=s_cfg['model_template'],
        specind_model=s_cfg['spectral_index_model'],
        curvature_index=float(s_cfg['curvature_index']),
        curvature_ref_freq=float(s_cfg['curvature_reference_freq'])
    )
    total += cube
    if utils.ConfigGetBoolean(cfg, param_file, "Synchrotron", 'save_cube'):
        write_cube("synch", cube)

# Free-Free
if utils.ConfigGetBoolean(cfg, param_file, "FreeFree", 'simulate'):
    f_cfg = utils.ConfigSectionMap(cfg, param_file, "FreeFree")
    cube = fg.free_free(
        frequencies, nside,
        model_template=f_cfg['model_template'],
        temp_electron=float(f_cfg['temp_electron'])
    )
    total += cube
    if utils.ConfigGetBoolean(cfg, param_file, "FreeFree", 'save_cube'):
        write_cube("free_free", cube)

# AME
if utils.ConfigGetBoolean(cfg, param_file, "AME", 'simulate'):
    a_cfg = utils.ConfigSectionMap(cfg, param_file, "AME")
    cube = fg.ame(
        frequencies, nside,
        model_template=a_cfg['model_template'],
        ame_ratio=float(a_cfg['ame_ratio']),
        ame_freq_in=float(a_cfg['ame_freq_in'])
    )
    total += cube
    if utils.ConfigGetBoolean(cfg, param_file, "AME", 'save_cube'):
        write_cube("ame", cube)

# Thermal Dust
if utils.ConfigGetBoolean(cfg, param_file, "ThermalDust", 'simulate'):
    t_cfg = utils.ConfigSectionMap(cfg, param_file, "ThermalDust")
    cube = fg.thermal_dust(
        frequencies, nside,
        template_model=t_cfg['model_template'],
        specind_model=t_cfg['spectral_index_model'],
        temp_model=t_cfg['temp_model']
    )
    total += cube
    if utils.ConfigGetBoolean(cfg, param_file, "ThermalDust", 'save_cube'):
        write_cube("thermal_dust", cube)

# Point Sources
if utils.ConfigGetBoolean(cfg, param_file, "PointSources", 'simulate'):
    p_cfg = utils.ConfigSectionMap(cfg, param_file, "PointSources")
    cube = fg.point_sources(
        frequencies, nside,
        model=p_cfg['model_source_count'],
        flux_poisson=float(p_cfg['max_flux_poisson_cl']),
        flux_max=float(p_cfg['max_flux_point_sources']),
        alpha=float(p_cfg['spectral_index']),
        sigma_alpha=float(p_cfg['spectral_index_std']),
        add_clustering=utils.ConfigGetBoolean(cfg, param_file, "PointSources", 'add_clustering')
    )
    total += cube
    if utils.ConfigGetBoolean(cfg, param_file, "PointSources", 'save_cube'):
        write_cube("point_sources", cube)

# CMB
if utils.ConfigGetBoolean(cfg, param_file, "CMB", 'simulate'):
    c_cfg = utils.ConfigSectionMap(cfg, param_file, "CMB")
    cube = fg.cmb(frequencies, nside, model=c_cfg['cmb_model'])
    total += cube
    if utils.ConfigGetBoolean(cfg, param_file, "CMB", 'save_cube'):
        write_cube("cmb", cube)

# Save total foregrounds
write_cube("foreground", total)

