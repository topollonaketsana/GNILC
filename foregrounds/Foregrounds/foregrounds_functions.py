import numpy as np
import healpy as hp
from scipy import interpolate, integrate
from astropy.io import fits as pyfits
import os

# Physical constants
k_bol = 1.38064852e-23  # J/K
h_planck = 6.62607004e-34  # m^2 kg / s
c = 299792458.0  # m/s
jy_to_si = 1e-26  # Jy to SI

__all__ = [
    "synchrotron", "free_free", "ame", "thermal_dust", "cmb",
    "point_sources", "temp_background_fixed_freq", "cl_poisson_fixed_freq",
    "map_poisson_fixed_freq_high_flow", "cl_cluster_fixed_freq", "source_count_battye"
]

def synchrotron(freqs, nside, template_model='haslam2014', specind_model='uniform', curvature_index=0.0, curvature_ref_freq=1.0):
    if template_model != 'haslam2014':
        raise ValueError("Only 'haslam2014' template supported")
    synch_map = hp.read_map('data/haslam408_dsds_Remazeilles2014_ns2048.fits')
    synch_map = hp.ud_grade(synch_map, nside_out=nside)

    # Corrigir pixels inválidos no mapa Haslam
    invalid = ~np.isfinite(synch_map) | (synch_map == hp.UNSEEN)
    valid = ~invalid
    if np.any(invalid):
        print(f"Interpolating {np.sum(invalid)} invalid pixels in synch_map...")
        theta, phi = hp.pix2ang(nside, np.arange(len(synch_map)))
        interp = interpolate.NearestNDInterpolator(np.vstack([theta[valid], phi[valid]]).T, synch_map[valid])
        synch_map[invalid] = interp(theta[invalid], phi[invalid])

    specind_path = {
        'uniform': 'data/synchrotron_specind_uniform.fits',
        'mamd2008': 'data/synchrotron_specind_mamd2008.fits',
        'giardino2002': 'data/synchrotron_specind_giardino2002.fits'
    }
    if specind_model not in specind_path:
        raise ValueError(f"Unknown spectral index model: {specind_model}")

    specind_map = hp.read_map(specind_path[specind_model])
    specind_map = hp.ud_grade(specind_map, nside_out=nside)

    freq_0 = 0.408
    npix = hp.nside2npix(nside)
    maps = np.zeros((len(freqs), npix), dtype=np.float32)

    for i, f in enumerate(freqs):
        exponent = specind_map - 2.0 + curvature_index * np.log10(f / curvature_ref_freq)
        maps[i] = 1e3 * (f / freq_0)**exponent * synch_map

    return maps

# --------------------------------------------------------------------------------
# Synchrotron
# --------------------------------------------------------------------------------

#def synchrotron(freqs, nside, template_model='haslam2014', specind_model='uniform', curvature_index=0.0, curvature_ref_freq=1.0):
#    if template_model != 'haslam2014':
#        raise ValueError("Only 'haslam2014' template supported")
#    synch_map = hp.read_map('data/haslam408_dsds_Remazeilles2014_ns2048.fits')
#    synch_map = hp.ud_grade(synch_map, nside_out=nside)
#
#    specind_path = {
#        'uniform': 'data/synchrotron_specind_uniform.fits',
#        'mamd2008': 'data/synchrotron_specind_mamd2008.fits',
#        'giardino2002': 'data/synchrotron_specind_giardino2002.fits'
#    }
#    if specind_model not in specind_path:3
#        raise ValueError(f"Unknown spectral index model: {specind_model}")
#
#    specind_map = hp.read_map(specind_path[specind_model])
#    specind_map = hp.ud_grade(specind_map, nside_out=nside)
#
#    freq_0 = 0.408
#    npix = hp.nside2npix(nside)
#    maps = np.zeros((len(freqs), npix), dtype=np.float32)
#
#    for i, f in enumerate(freqs):
#        exponent = specind_map - 2.0 + curvature_index * np.log10(f / curvature_ref_freq)
#        maps[i] = 1e3 * (f / freq_0)**exponent * synch_map
#
#    return maps

# --------------------------------------------------------------------------------
# Free-Free
# --------------------------------------------------------------------------------

def free_free(freqs, nside, model_template='dickinson2003', temp_electron=7000.0):
    if model_template != 'dickinson2003':
        raise ValueError("Only 'dickinson2003' model supported")
    halpha = hp.read_map('data/onedeg_diff_halpha_JDprocessed_smallscales_2048.fits')
    halpha = hp.ud_grade(halpha, nside_out=nside)

    maps = np.zeros((len(freqs), hp.nside2npix(nside)), dtype=np.float32)
    a_factor = 0.366 * freqs**0.1 * temp_electron**(-0.15) * (np.log(4.995e-2 / freqs) + 1.5 * np.log(temp_electron))

    for i, f in enumerate(freqs):
        maps[i] = 8.396 * a_factor[i] * f**(-2.1) * (temp_electron * 1e-4)**0.667 * 10**(0.029 / (temp_electron * 1e-4)) * 1.08 * halpha

    return maps

# --------------------------------------------------------------------------------
# AME
# --------------------------------------------------------------------------------

def ame(freqs, nside, model_template='planck_t353', ame_ratio=1.0, ame_freq_in=30.0):
    if model_template != 'planck_t353':
        raise ValueError("Only 'planck_t353' model supported")

    tau = hp.read_map('data/Planck_map_t353_small_scales.fits')
    tau = hp.ud_grade(tau, nside_out=nside)

    spdust_file = 'data/spdust2_cnm.dat'
    f_table, flux_table = np.loadtxt(spdust_file, comments=';', unpack=True)
    flux_interp = interpolate.interp1d(f_table, flux_table, fill_value="extrapolate")

    flux_ref = flux_interp(ame_freq_in)
    maps = np.zeros((len(freqs), hp.nside2npix(nside)), dtype=np.float32)

    for i, f in enumerate(freqs):
        scale = flux_interp(f) / flux_ref * (ame_freq_in / f)**2
        maps[i] = 1e-3 * tau * ame_ratio * scale

    return maps

# --------------------------------------------------------------------------------
# Thermal Dust
# --------------------------------------------------------------------------------

def thermal_dust(freqs, nside, template_model='gnilc_353', specind_model='gnilc_353', temp_model='gnilc_353'):
    if template_model != 'gnilc_353':
        raise ValueError("Only 'gnilc_353' dust model supported")

    tau = hp.read_map('data/COM_CompMap_Dust-GNILC-F353_2048_R2.00_small_scales.fits')
    beta = hp.read_map('data/COM_CompMap_Dust-GNILC-Model-Spectral-Index_2048_R2.00.fits')
    temp = hp.read_map('data/COM_CompMap_Dust-GNILC-Model-Temperature_2048_R2.00.fits')

    tau = 261.20305067644796 * hp.ud_grade(tau, nside_out=nside)
    beta = hp.ud_grade(beta, nside_out=nside)
    temp = hp.ud_grade(temp, nside_out=nside)

    maps = np.zeros((len(freqs), hp.nside2npix(nside)), dtype=np.float32)
    gamma = h_planck / (k_bol * temp)
    freq_ref = 353.0

    for i, f in enumerate(freqs):
        num = np.exp(gamma * freq_ref * 1e9) - 1
        den = np.exp(gamma * f * 1e9) - 1
        maps[i] = 1e-3 * tau * (f / freq_ref)**(beta + 1) * (num / den)

    return maps

# --------------------------------------------------------------------------------
# CMB
# --------------------------------------------------------------------------------

def cmb(freqs, nside, model='standard'):
    if model != 'standard':
        raise ValueError("Only 'standard' model is supported")

    cl_tt = np.loadtxt('data/cmb_tt.txt')
    ell = np.loadtxt('data/cmb_ell.txt')
    lmax = 3 * 2048 - 1
    cl = np.zeros(lmax + 1)

    for l in range(len(cl_tt)):
        cl[l + 2] = 2 * np.pi * cl_tt[l] / (ell[l] * (ell[l] + 1))

    cmb_map = 1e-6 * hp.synfast(cl, nside, verbose=False)
    gamma = h_planck / k_bol
    T_cmb = 2.73

    maps = np.zeros((len(freqs), hp.nside2npix(nside)), dtype=np.float32)
    for i, f in enumerate(freqs):
        x = gamma * f * 1e9 / T_cmb
        maps[i] = (x**2 * np.exp(x) / (np.exp(x) - 1)**2) * cmb_map

    return 1e3 * maps

# --------------------------------------------------------------------------------
# Source Count and Point Sources
# --------------------------------------------------------------------------------

def source_count_battye(flux):
    if flux <= 1.0:
        logS = np.log10(flux)
        coeffs = [2.593, 9.333e-2, -4.839e-4, 2.488e-1, 8.995e-2, 8.506e-3]
        poly = sum(c * logS**i for i, c in enumerate(coeffs))
        return flux**-2.5 * 10**poly
    else:
        a, b = 2.59300238, -2.40632446
        return 10**(a + b * np.log10(flux))


def temp_background_fixed_freq(flux_max, model='battye2013'):
    if model != 'battye2013':
        raise ValueError("Only 'battye2013' model supported")

    fixed_freq = 1.4e9
    flux_min = 1e-6
    conv = 2 * k_bol * (fixed_freq / c)**2
    s = np.logspace(np.log10(flux_min), np.log10(flux_max), 10000)
    integrand = s * np.vectorize(source_count_battye)(s)
    temp_ps = jy_to_si / conv * integrate.simpson(integrand, s)

    return 1e3 * temp_ps


def cl_poisson_fixed_freq(flux_max, model='battye2013'):
    if model != 'battye2013':
        raise ValueError("Only 'battye2013' model supported")

    fixed_freq = 1.4e9
    flux_min = 1e-6
    conv = 2 * k_bol * (fixed_freq / c)**2
    s = np.logspace(np.log10(flux_min), np.log10(flux_max), 10000)
    integrand = s**2 * np.vectorize(source_count_battye)(s)
    aps = (jy_to_si / conv)**2 * integrate.simpson(integrand, s)

    return 1e6 * aps


def map_poisson_fixed_freq_high_flow(flux_min, flux_max, nside, model='battye2013'):
    if model != 'battye2013':
        raise ValueError("Only 'battye2013' model supported")

    npix = hp.nside2npix(nside)
    omega_pix = 4 * np.pi / npix
    map_out = np.zeros(npix)
    decades = int(np.floor(np.log10(flux_max / flux_min)))
    conv = 2 * k_bol * (1.4e9 / c)**2

    for d in range(decades):
        s0 = flux_min * 10**d
        s1 = s0 * 10
        s = np.linspace(s0, s1, 100)
        dn_ds = np.vectorize(source_count_battye)(s)
        delta_n = int(np.round(integrate.simpson(dn_ds, s)))
        for _ in range(delta_n):
            flux = np.random.uniform(s0, s1)
            temp = jy_to_si / conv / omega_pix * flux
            pix = np.random.randint(0, npix)
            map_out[pix] += temp

    return 1e3 * map_out


def cl_cluster_fixed_freq(ell, t_ps, model='battye2013'):
    if model != 'battye2013':
        raise ValueError("Only 'battye2013' model supported")
    ell = np.maximum(ell, 1)
    return 1.8e-4 * ell**-1.2 * t_ps**2


def point_sources(freqs, nside, model='battye2013', flux_poisson=1e-3, flux_max=1e1, alpha=-0.7, sigma_alpha=0.3, add_clustering=False):
    lmax = 3 * nside - 1
    npix = hp.nside2npix(nside)
    nch = len(freqs)
    maps = np.zeros((nch, npix))

    cl_poiss = cl_poisson_fixed_freq(flux_poisson, model)
    cl_arr = np.ones(lmax + 1) * cl_poiss
    map_poisson = hp.synfast(cl_arr, nside, lmax=lmax, verbose=False)
    t_ps = temp_background_fixed_freq(flux_max, model)

    if add_clustering:
        cl_cluster = np.array([cl_cluster_fixed_freq(l, t_ps, model) for l in range(lmax + 1)])
        map_cluster = hp.synfast(cl_cluster, nside, lmax=lmax, verbose=False)
    else:
        map_cluster = 0

    high_flux_map = map_poisson_fixed_freq_high_flow(flux_poisson, flux_max, nside, model)
    alpha_map = np.random.normal(alpha, sigma_alpha, npix)

    for i, f in enumerate(freqs):
        scale = (f / 1.4)**alpha_map
        maps[i] = scale * (map_poisson + map_cluster + high_flux_map + t_ps)

    return maps

