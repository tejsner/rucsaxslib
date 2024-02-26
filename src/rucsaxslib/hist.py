'''
Azimuthal integration of images
'''
import numpy as np


def integrate(img, x='q', mask=None, bins=500, range=None):
    # allways mask invalid pixels
    if mask is None:
        mask = ~np.isnan(img.data)
    elif (isinstance(mask, np.ndarray) and
          (mask.dtype == 'bool') and
          (mask.shape == img.data.shape)):
        mask = np.logical_and(mask, ~np.isnan(img.data))
    else:
        raise ValueError()

    if isinstance(x, str):
        if x == 'q':
            x = img.get_q()
        else:
            raise ValueError()

    # get total intensity in each bin
    y_total, x_edges = np.histogram(x[mask], weights=img.data[mask],
                                    bins=bins, range=range)
    # get nummber of pixels in each bin
    y_npix, _ = np.histogram(x[mask],
                             bins=bins, range=range)

    # mask histogram bins lying between x-values (typically pixels
    # centers). This can happen with fine-grained binning. This mask
    # is applied to all return values.
    hmask = np.where(y_npix != 0)

    # we want intensity per pixel for proper normalization
    y = y_total[hmask]/y_npix[hmask]

    # errors are added in quadrature, so the error is sqrt(sum(error**2))
    variance, _ = np.histogram(x[mask], weights=img.error[mask]**2,
                               bins=bins, range=range)
    error_total = np.sqrt(variance)
    error = error_total[hmask]/y_npix[hmask]

    # return x-axis as bin centers
    # TODO: is this correct for log-scaled bins?
    x = x_edges[:-1] + np.diff(x_edges)/2

    return x[hmask], y, error
