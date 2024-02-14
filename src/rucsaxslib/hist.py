'''
Azimuthal integration of images
'''
import numpy as np


def integrate(img, x=None, mask=None, **kwargs):
    if mask is None:
        mask = ~np.isnan(img.data)

    if x is None or x == 'q':
        x = img.get_q()

    default_kwargs = {'bins': 500}
    if kwargs:
        kwargs = {**default_kwargs, **kwargs}
    else:
        kwargs = default_kwargs

    y_total, x_edges = np.histogram(x[mask], weights=img.data[mask], **kwargs)
    y_npix, _ = np.histogram(x[mask], **kwargs)
    y_total[y_total == 0] = np.nan
    y = y_total/y_npix

    variance, _ = np.histogram(x[mask], weights=img.error[mask]**2, **kwargs)
    error_total = np.sqrt(variance)
    error = error_total/y_npix

    x = x_edges[:-1] + np.diff(x_edges)/2

    return x, y, error
