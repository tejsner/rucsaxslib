# rucsaxslib
`rucsaxslib` is a Python package intended for treatment of 2D X-ray scattering data.
While the library is designed to be generally applicable, it has only been tested on data from the [Xeuss3](https://www.xenocs.com/saxs-products/saxs-equipment-xeuss) installed at Roskilde University (RUC).

While I consider the library in a functional state, its primary purpose is to clarify how data reduction is done in practice.
Image operations are performed only using the standard Python library and [NumPy](https://numpy.org/).
I thus recommend reading through the code, in partcular [img.py](src/rucsaxslib/img.py).

The code has not been extensively tested, so expect bugs.

## Contents

- [Installation](#installation)
  - [Manual installation](#manual-installation)
  - [From TestPyPI](#from-testpypi)
- [Overview](#overview)
- [Detector Data](#detector-data)
- [Plaintext Data](#plaintext-data)
- [Integration](#integration)
- [GISAXS](#gisaxs)
- [Reflectivity](#reflectivity)
- [STVA](#stva)
- [Header keywords](#header-keywords)

## Installation

### Manual installation

Clone this respository and run `pip install -e rucsaxslib` in an appropriate virtual environment.

### From TestPyPI

Currently resides at <https://test.pypi.org/project/rucsaxslib/>.

```
pip install numpy matplotlib fabio
pip install --index-url https://test.pypi.org/simple/ --no-deps rucsaxslib
```

## Overview

`rucsaxslib` exposes the following functions on import:

- `rucsaxslib.ImgData`: Class that holds detector data
- `rucsaxslib.from_rucsaxs`: Read an image file from Xeuss3 and return an `ImgData` object.
- `rucsaxslib.integrate`: Integrate an image onto a single axis (typically $q$ or $2\theta$)
- `rucsaxslib.gi_integrate`: Integrate GISAXS images along $q_{xy}$ or $q_z$
- `rucsaxslib.gi_draw_bbox`: Draw the bounding box used by `gi_integrate`.
- `rucsaxslib.reflectivity`: Get reflectivity curve from a list of images.
- `rucsaxslib.stva`: Functions to parse data from the Solvent Thermal Vapor Annealing (STVA) setup used at RUC.
- `rucsaxslib.xeuss`: Functions to parse plaintext data generated by Xeuss3.

Run `help()` on any of these objects from a Python prompt for details.

## Detector Data
Detector data is stored in `ImgData` objects.
`ImgData` objects are initialized by passing the raw detector counts (numpy array) and header information (dictionary).
In order for data reduction to work properly, there are a number of keys in the header that are required (such as sample-detector distance).
These keywords are listed in [Header keywords](#header-keywords)

When working with data from the rucsaxs laboratory, the helper function `from_rucsaxs` takes care of loading the image file and setting the required keywords.
In practice, one loads an image (in this case a file named `data.edf`) by running the following:

```python
import rucsaxslib as rs
img = rs.from_rucsaxs('data.edf')
```

This loads the raw image data that can be accessed from the `img.data` attribute.
Header information is saved in the `img.header` attribute and the standard error for each pixel is saved in the `img.error` attribute.
To perform all standard corrections to the pixel intensities (time, flux, transmission, solid angle and polarization), run

```python
import rucsaxslib as rs
img = rs.from_rucsaxs('data.edf', corrs='all')
```

`ImgData` objects contain a number of useful methods:

- `ImgData.apply_corrections`: Corrects data and error in order to the differential scattering cross-section for each pixel.
  As shown above, this can be run when initializing the object through the `corrs` keyword argument.
- `ImgData.get_coordinates`: Get coordinates of each pixel in various reference systems.
- `ImgData.get_q`: Get the magnitude of the reciprocal wavevector $q$ (in inverse angstrom) for each pixel.
- `ImgData.get_tth`: Get the scattering angle $2\theta$ (in degrees) for each pixel.
- `ImgData.plot`: Plot the detector image using [Matplotlib](https://matplotlib.org/).

## Plaintext Data
In addition to image files in `.edf` format, the Xeuss3 instrument also has output data in plaintext format.
These files are

- `.dat`: Azimuthally integrated data.
- `.dat` (BioCube): Various azimuthally integrated datasets generated by the BioCube software.
  This includes the raw azimuthally integrated date, but also the result of operations such as averaging frames and buffer subtraction.
- `.log`: Log file from SPEC. These contains all the scans performed using SPEC.

`rucsaxslib` includes the following convenience functions for loading these types of data:

- `rucsaxslib.xeuss.read_dat`: Read standard `.dat` files.
- `rucsaxslib.xeuss.read_dat_bio`: Read BioCube `.dat` files.
- `rucsaxslib.xeuss.read_spec`: Read SPEC `.log` files.

## Integration
To integrate an image along a given axis, `rucsaxslib` includes the `integrate` function.
This function is implemented using `numpy.histogram` and has not been optimized for performance.
To azimuthally integrate an image (`data.edf`), one could do the following

```python
import rucsaxslib as rs
img = rs.from_rucsass('data.edf', corrs='all')
q, I, err = rs.integrate(img, x=img.get_q(), bins=500, xrange=(0, 0.5))
```

Or equivalently (using the shorthand "q" as the argument for the x-axis):

```python
import rucsaxslib as rs
img = rs.from_rucsass('data.edf', corrs='all')
q, I, err = rs.integrate(img, x='q', bins=500, xrange=(0, 0.5))
```

This will load the image, perform all corrections, integrate with 500 bins between 0 and 0.5 inverse angstrom.
If the `bins` argument is omitted, the procedure will default to 500 bins.
If the `xrange` argument is omitted, the range will be the minimum and maximum of the data.

## GISAXS
When doing GISAXS (Grazing Incidence Small-Angle X-ray Scattering), it can be useful to integrate images along the $q_{xy}$ and $q_z$ axes separately.
This can be achieved using the `get_coordinates` method of an `ImgData` object along with the `rucsaxslib.integrate` function:

```python
import rucsaxslib as rs
img = rs.from_rucsass('data.edf', corrs=['PO', 'SP'])
img_qxy, img_qz = img.get_coordinates("gisaxs")
qxy, I, err = rs.integrate(img, x=img_qxy, bins=500, xrange=(-0.2, 0.2))
```

Which will integrate the image along $q_{xy}$ between -0.2 and 0.2 inverse angstrom.
To simplify this operation, `rucsaxslib` includes the function `gi_integrate`.
The following code is equivalent to the above snippet:

```python
import rucsaxslib as rs
img = rs.from_rucsass('data.edf', corrs=['PO', 'SP'])
qxy, I, err = rs.gi_integrate(img, axis='qxy', bins=500, qxylim=(-0.2, 0.2))
```

In addition, this function also makes it simpler to restrict the integration area in both directions, i.e. the following

```python
import rucsaxslib as rs
img = rs.from_rucsass('data.edf', corrs=['PO', 'SP'])
qxy, I, err = rs.gi_integrate(img, axis='qxy', bins=500, qxylim=(-0.2, 0.2), qzlim=(0, 0.3))
```

will integrate in the $q_{xy}$ direction from -0.2 to 0.2  while restricting the integration in the $q_z$ direction to being between 0 and 0.3.

To visualize the area integrated, one can use the function `gi_draw_bbox` which will draw a bounding box on a matplotlib axis.
This takes the same `qxylim` and `qzlim` arguments as `gi_integrate`, so one can use these functions like so:

```python
import rucsaxslib as rs
import matplotlib.pyplot as plt

img = rs.from_rucsass('data.edf', corrs=['PO', 'SP'])
qxylim = (-0.2, 0.2)
qzlim = (0, 0.3)
qxy, I, err = rs.gi_integrate(img, axis='qxy', bins=200, qxylim=qxylim, qzlim=qzlim)

f, ax = plt.subplots(ncols=2)
img.plot(ax=ax[0], coords='gisaxs', rebin=300)
rs.gi_draw_bbox(ax[0], qxylim, qzlim)
ax[1].errorbar(qxy, I, err)
```

## Reflectivity
**Warning: Not thoroughly tested. Use with caution.**

The function `reflectivity` contains some rudimentary functionality to produce reflectivity data from raw images.
`rucsaxslib.reflectivity` expects a list of `.edf` filenames that are a result of reflectivity scan with increasing value of incident angle. If one has a folder with a reflectivity scan, one can use this function in the following way:

```python
import rucsaxslib as rs
import glob

filenames = sorted(glob.glob('reflectivity/*.edf'))
angle, I, err = rs.reflectivity(filenames, roi_size=(21, 21), angle_offset=0)
```

The optional keyword arguments can be used to adjust the region of interest and an angle offset (in case the experiment is not properly aligned).

## STVA
**Warning: In very early stage and not thoroughly tested. Use with caution.**

The `rucsaxslib.stva` module contains a number of functions to handle data generated by the STVA setup used at RUC.
It contains the following functions:

- `rucsaxslib.stva.read_nanocalc_log`: Read log file from the Nanocalc software.
- `rucsaxslib.stva.read_flowplot_log`: Read log file from the FlowPlot software.
- `rucsaxslib.stva.read_svc_log`: Read log file from the SVC box.
- `rucsaxslib.stva.read_gisaxs_log`: Get list of GISAXS images and their timestamps from a folder.
- `rucsaxslib.stva.merge_gi_datasets`: Merge datasets generated from the previous 4 functions.

## Header keywords

See [Boesecke (2007)](https://doi.org/10.1107/S0021889807001100) for details.

| Keyword              | Required           | Default  | Type    | Description                                                       |
|----------------------|--------------------|----------|---------|-------------------------------------------------------------------|
| `Dim_1`              | :white_check_mark: |          | `int`   | Number of pixels along axis 1                                     |
| `Dim_2`              | :white_check_mark: |          | `int`   | Number of pixels along axis 2                                     |
| `RasterOrientation`  |                    | 3        | `int`   | Raster orientation number                                         |
| `Offset_1`           |                    | 0        | `int`   | Spatial array offset along axis 1                                 |
| `Offset_2`           |                    | 0        | `int`   | Spatial array offset along axis 2                                 |
| `BSize_1`            |                    | 1        | `int`   | Pixel size relative to size of unbinned pixel along axis 1        |
| `BSize_2`            |                    | 1        | `int`   | Pixel size relative to size of unbinned pixel along axis 2        |
| `PSize_1`            |                    | 7.5e-5   | `float` | Pixel size in meters along axis 1                                 |
| `PSize_2`            |                    | 7.5e-5   | `float` | Pixel size in meters along axis 2                                 |
| `Center_1`           | :white_check_mark: |          | `float` | Point of Normal Incidence (axis 1)                                |
| `Center_2`           | :white_check_mark: |          | `float` | Point of Normal Incidence (axis 2)                                |
| `SampleDistance`     | :white_check_mark: |          | `float` | Sample-detector distance in meters                                |
| `WaveLength`         |                    | 1.54e-10 | `float` | X-ray wavelength in meters                                        |
| `IncidentAngle`      |                    | 0        | `float` | Incident angle (used for GISAXS)                                  |
| `Dummy`              |                    |          | `float` | Dummy value                                                       |
| `DDummy`             |                    |          | `float` | Range around Dummy                                                |
| `Time`               |                    |          | `str`   | Start time of exposure in ISO8601                                 |
| `Title`              |                    |          | `str`   | Title string                                                      |
| `ExposureTime`       |                    | 1.0      | `float` | Exposure time                                                     |
| `DarkConstant`       |                    | 0.0      | `float` | Dark constant subtracted from all pixels                          |
| `TransmittedFlux`    |                    | 1.0      | `float` | Flux of transmitted beam                                          |
| `SourcePolarization` |                    | 0.0      | `float` | Polarization of source (-1 horizontal, 0 unpolarized, 1 vertical) |
