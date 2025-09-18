# function for testing

import numpy as np
import healpy as hp

def free_free(freq_GHz, nside=256, temp_electron=8000.0, **kwargs):
    """
    Generate free-free emission map
    
    Parameters
    ----------
    freq_GHz : float or array-like
        Frequency in GHz
    nside : int
        HEALPix nside parameter
    temp_electron : float
        Electron temperature in K
    
    Returns
    -------
    map : ndarray
        Free-free temperature map in mK
    """
    beta = -2.1  # spectral index
    A_ff = 10.0  # amplitude in mK at reference frequency
    freq_ref = 1.0  # reference frequency in GHz
    
    npix = hp.nside2npix(nside)
    
    # Generate template map
    template = np.random.randn(npix)
    template = hp.smoothing(template, fwhm=np.radians(1.0))
    
    # Scale with frequency
    if hasattr(freq_GHz, '__len__'):
        output = np.zeros((len(freq_GHz), npix))
        for i, freq in enumerate(freq_GHz):
            output[i] = A_ff * template * (freq/freq_ref)**beta
        return output
    else:
        return A_ff * template * (freq_GHz/freq_ref)**beta