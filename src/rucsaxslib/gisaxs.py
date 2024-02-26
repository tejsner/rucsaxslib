'''
GISAXS and reflectometry
'''
import numpy as np
from .hist import integrate
from .img import from_rucsaxs
import matplotlib.patches as patches


def gi_integrate(img, axis='qxy', bins=100,
                 qxylim=(None, None), qzlim=(None, None)):
    '''
    Perform GISAXS line integration
    '''
    qxy, qz = img.get_coordinates("gisaxs")

    # initialize mask with all pixels active
    mask = np.ones_like(img.data, dtype=bool)

    if qxylim[0] is not None:
        mask = np.logical_and(mask, qxy >= qxylim[0])
    if qxylim[1] is not None:
        mask = np.logical_and(mask, qxy <= qxylim[1])
    if qzlim[0] is not None:
        mask = np.logical_and(mask, qz >= qzlim[0])
    if qzlim[1] is not None:
        mask = np.logical_and(mask, qz <= qzlim[1])

    if axis == 'qxy':
        x, y, error = integrate(img, x=qxy, mask=mask,
                                xrange=qxylim, bins=bins)
    elif axis == 'qz':
        x, y, error = integrate(img, x=qz, mask=mask,
                                xrange=qzlim, bins=bins)
    else:
        raise ValueError('axis argument must be either "qxy" or "qz"')

    return x, y, error


def gi_draw_bbox(ax, qxylim, qzlim, **kwargs):
    '''
    draw a bounding box used for line integration on figure axes.
    Assumes figure with qxy horizontal and qz vertical.

    Simple wrapper for matplotlib.patches.Rectangle that works with
    the arguments used for line integration.

    Extra keyword arguments are passed to Rectangle().
    '''
    xy = (qxylim[0], qzlim[0])
    width = qxylim[1] - qxylim[0]
    height = qzlim[1] - qzlim[0]

    default_kwargs = {'edgecolor': 'r', 'facecolor': 'none'}
    if kwargs:
        kwargs = {**default_kwargs, **kwargs}
    else:
        kwargs = default_kwargs

    rect = patches.Rectangle(xy, width, height, **kwargs)
    ax.add_patch(rect)


def reflectivity(filenames, roi_size=(21, 21), angle_offset=0):
    '''
    Get reflectivity curve given a list of files.
    Assumes raster configuration 3 or 4 (compatible with RUCSAXS).
    Assumes files sorted by ascending value of incident angle.
    '''
    # load data
    imgs = [from_rucsaxs(fn) for fn in filenames]

    # get needed metadata from first image
    sd = imgs[0].header['SampleDistance']
    p2 = imgs[0].header["PSize_2"]
    c1, c2 = int(imgs[0].header["Center_1"]), int(imgs[0].header["Center_2"])

    x, y, e = [], [], []

    for img in imgs:
        alpha_i = (img.header['IncidentAngle'] + angle_offset)*np.pi/180
        Lp_px = np.sin(2*alpha_i)*sd/p2  # distance in pixels to reflected beam
        rz = round(c2 - Lp_px)  # index of (expected) reflected pixel

        # pixel limits for bounding box
        x_pixels = (c1 - roi_size[0]//2, c1 + roi_size[0]//2)
        z_pixels = (rz - roi_size[1]//2, rz + roi_size[1]//2)

        # integrate pixels in roi
        s = np.sum(img.data[z_pixels[0]:z_pixels[1]+1,
                            x_pixels[0]:x_pixels[1]+1])

        # get variance in roi
        v = np.sum(img.error[z_pixels[0]:z_pixels[1]+1,
                             x_pixels[0]:x_pixels[1]+1]**2)

        x.append(img.header['IncidentAngle'])
        y.append(s)
        e.append(np.sqrt(v))

    return x, y, e
