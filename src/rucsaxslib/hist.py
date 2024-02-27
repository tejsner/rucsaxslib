'''
Integration of 2d images onto any supplied axis.
Typically used for azimuthal integration.
'''
import numpy as np


def integrate(img, x='q', mask=None, bins=500, xrange=None):
    # Allways mask np.nan pixels explictely in addition to the
    # (optionally) user-defined mask. This avoids over-counting in the
    # denominator (y_npix).
    if mask is None:
        mask = ~np.isnan(img.data)
    elif (isinstance(mask, np.ndarray) and
          (mask.dtype == 'bool') and
          (mask.shape == img.data.shape)):
        mask = np.logical_and(mask, ~np.isnan(img.data))
    else:
        raise ValueError("Mask must be numpy.ndarray with dtype bool and have"
                         "the same shape as data")

    # get x-axis automatically for certain string arguments.
    if isinstance(x, str):
        if x == 'q':
            x = img.get_q()
        elif x == 'tth':
            x = img.get_tth()
        else:
            raise ValueError('String argument "{}" for x-axis not recognized'.
                             format(x))

    # handle range arguments
    if xrange is None:
        xrange = [x.min(), x.max()]
    elif ((isinstance(xrange, tuple) or isinstance(xrange, list))
          and len(xrange) == 2):
        xrange = list(xrange)
        if xrange[0] is None:
            xrange[0] = x.min()
        if xrange[1] is None:
            xrange[1] = x.max()
    else:
        raise ValueError("Range argument must be None or"
                         "list with [xmin, xmax]")

    # get total intensity in each bin
    y_total, x_edges = np.histogram(x[mask], weights=img.data[mask],
                                    bins=bins, range=xrange)
    # get nummber of pixels in each bin
    y_npix, _ = np.histogram(x[mask],
                             bins=bins, range=xrange)

    # mask histogram bins lying between x-values (typically pixels
    # centers). This can happen with fine-grained binning. This mask
    # is applied to all return values.
    hmask = np.where(y_npix != 0)

    # we want intensity per pixel for proper normalization
    y = y_total[hmask]/y_npix[hmask]

    # errors are added in quadrature, so the error is sqrt(sum(error**2))
    variance, _ = np.histogram(x[mask], weights=img.error[mask]**2,
                               bins=bins, range=xrange)
    error_total = np.sqrt(variance)
    error = error_total[hmask]/y_npix[hmask]

    # return x-axis as bin centers
    # TODO: is this correct for log-scaled bins?
    x = x_edges[:-1] + np.diff(x_edges)/2

    return x[hmask], y, error
