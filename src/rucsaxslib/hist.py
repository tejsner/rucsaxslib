'''
Azimuthal integration of images
'''
import numpy as np


def integrate(img, coords='q', bins=500, xrange=None):
    mask = ~np.isnan(img.data)

    if coords == 'q':
        x = img.get_q()
    elif coords == 'tth':
        x, _ = img.get_coordinates(reference_system='polar')

    y_total, x_edges = np.histogram(x[mask],
                                    bins=bins,
                                    range=xrange,
                                    weights=img.data[mask])

    y_npix, _ = np.histogram(x[mask],
                             range=xrange,
                             bins=bins)

    y_total[y_total == 0] = np.nan
    y = y_total/y_npix

    # error
    variance, _ = np.histogram(x[mask],
                               bins=bins,
                               range=xrange,
                               weights=img.error[mask]**2)
    error_total = np.sqrt(variance)
    error = error_total/y_npix

    # get x values (bin centers)
    x = x_edges[:-1] + np.diff(x_edges)/2

    return x, y, error
