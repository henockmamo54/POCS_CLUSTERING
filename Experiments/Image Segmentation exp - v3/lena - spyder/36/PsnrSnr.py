import numpy as np
def mse(vref, vcmp):
    """
    Compute Mean Squared Error (MSE) between two images.

    Parameters
    ----------
    vref : array_like
      Reference image
    vcmp : array_like
      Comparison image

    Returns
    -------
    x : float
      MSE between `vref` and `vcmp`
    """

    r = np.asarray(vref, dtype=np.float64).ravel()
    c = np.asarray(vcmp, dtype=np.float64).ravel()
    return np.mean(np.abs(r - c)**2)

def snr(vref, vcmp):
    """
    Compute Signal to Noise Ratio (SNR) of two images.

    Parameters
    ----------
    vref : array_like
      Reference image
    vcmp : array_like
      Comparison image

    Returns
    -------
    x : float
      SNR of `vcmp` with respect to `vref`
    """

    dv = np.var(vref)
    with np.errstate(divide='ignore'):
        rt = dv / mse(vref, vcmp)
    return 10.0 * np.log10(rt)




def psnr(vref, vcmp, rng=None):
    """
    Compute Peak Signal to Noise Ratio (PSNR) of two images. The PSNR
    calculation defaults to using the less common definition in terms
    of the actual range (i.e. max minus min) of the reference signal
    instead of the maximum possible range for the data type
    (i.e. :math:`2^b-1` for a :math:`b` bit representation).

    Parameters
    ----------
    vref : array_like
      Reference image
    vcmp : array_like
      Comparison image
    rng : None or int, optional (default None)
      Signal range, either the value to use (e.g. 255 for 8 bit samples) or
      None, in which case the actual range of the reference signal is used

    Returns
    -------
    x : float
      PSNR of `vcmp` with respect to `vref`
    """

    if rng is None:
        rng = vref.max() - vref.min()
    dv = (rng + 0.0)**2
    with np.errstate(divide='ignore'):
        rt = dv / mse(vref, vcmp)
    return 10.0 * np.log10(rt)


