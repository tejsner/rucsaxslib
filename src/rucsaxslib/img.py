'''
This module contains functionality related to image operations.

Implementation is done using the geometry outlined in
https://doi.org/10.1107/S0021889807001100
'''
import numpy as np
import fabio
import matplotlib
import matplotlib.pyplot as plt
import math

# Raster orientation conversion matrices
RASTER_MAT = {1: np.matrix(((1, 0), (0, 1))),
              2: np.matrix(((-1, 0), (0, 1))),
              3: np.matrix(((1, 0), (0, -1))),
              4: np.matrix(((-1, 0), (0, -1))),
              5: np.matrix(((0, 1), (1, 0))),
              6: np.matrix(((0, 1), (-1, 0))),
              7: np.matrix(((0, -1), (1, 0))),
              8: np.matrix(((0, -1), (-1, 0)))}

# key: keyword, val: (type, default value, category). for geometry keywords, a
# default value of None indicates that the keyword is required
KEYWORDS = {'Dim_1': (int, None, 'geometry'),
            'Dim_2': (int, None, 'geometry'),
            'RasterOrientation': (int, 3, 'geometry'),
            'Offset_1': (int, 0, 'geometry'),
            'Offset_2': (int, 0, 'geometry'),
            'BSize_1': (int, 1, 'geometry'),
            'BSize_2': (int, 1, 'geometry'),
            'PSize_1': (float, 7.5e-05, 'geometry'),
            'PSize_2': (float, 7.5e-05, 'geometry'),
            'Center_1': (float, None, 'geometry'),
            'Center_2': (float, None, 'geometry'),
            'SampleDistance': (float, None, 'geometry'),
            'WaveLength': (float, 1.541891e-10, 'geometry'),
            'IncidentAngle': (float, 0, 'geometry'),
            'DetectorRotation_1': (float, 0, 'geometry'),
            'DetectorRotation_2': (float, 0, 'geometry'),
            'DetectorRotation_3': (float, 0, 'geometry'),
            'ProjectionType': (str, 'saxs', 'geometry'),
            'Dummy': (float, None, 'optional'),
            'DDummy': (float, None, 'optional'),
            'Time': (str, None, 'optional'),
            'Title': (str, None, 'optional'),
            'Intensity0': (float, None, 'intensity'),
            'Intensity1': (float, None, 'intensity'),
            'ExposureTime': (float, 1, 'intensity'),
            'DarkConstant': (float, 0.0, 'intensity'),
            'TransmittedFlux': (float, 1.0, 'intensity'),
            'SourcePolarization': (float, 0, 'intensity')}

# axis labels for plotting
AX_LABELS = {'normal': {'x': 'distance [mm]',
                        'y': 'distance [mm]'},
             'wavevector': {'x': r'$q_x \; [Å^{-1}]$',
                            'y': r'$q_z \; [Å^{-1}]$'},
             'gisaxs': {'x': r'sgn$(q_x) \times q_{xy} \; [Å^{-1}]$',
                        'y': r'$q_z \; [Å^{-1}]$'},
             'polar': {'x': r'2$\theta$ [degrees]',
                       'y': r'$\phi$ [degrees]'}}


def from_file(filename, engine='fabio', header_rename={},
              header_extra={}, **kwargs):
    '''Load a detector image from a file

    Args:
        filename (str): path to filename.
        engine (str, optional): Loading engine. Defaults to and only supports
            "fabio".
        header_rename (dict, optional): Dictionary to rename header variables.
            Dictionary keys should be the current name, values should be the
            desired nams. Skips variables not present in the header.
        header_extra: (dict, optional): Extra header variables to add. If
            variable allready exists it will be overwritten.
        **kwargs: Additional keywords passed into the ImgData constructor.

    Returns:
        ImgData object containing the image data and header information.
    '''
    if engine == 'fabio':
        faimg = fabio.open(filename)

        for key in header_rename:
            if key in faimg.header:
                faimg.header[header_rename[key]] = faimg.header[key]

        for key in header_extra:
            faimg.header[key] = header_extra[key]

        return ImgData(faimg.data, faimg.header, **kwargs)
    else:
        raise NotImplementedError(f'Engine {engine} not implemented')


def from_rucsaxs(filename, **kwargs):
    '''Load a detector image from RUCSAXS.

    Specialized version of from_file that has default parameters appropriate
    for data acquired at the RUCSAXS instrument.

    Args:
        filename (str): Path to filename.
        **kwargs: Additional keywords passed into the ImgData constructor.

    Returns:
        ImgData object containing the image data and header information.
    '''
    rucsaxs_rename = {'Comment': 'Title',
                      'Date': 'Time',
                      'BackgroundCorrectionConstant': 'DarkConstant',
                      'om': 'IncidentAngle'}

    rucsaxs_extra = {'RasterOrientation': 3,
                     'SourcePolarization': 0}

    default_kwargs = {'engine': 'fabio',
                      'header_rename': rucsaxs_rename,
                      'header_extra': rucsaxs_extra,
                      'dark_subtracted': True,
                      'mask_le_dummy': True}

    kwargs = {**default_kwargs, **kwargs}
    img = from_file(filename, **kwargs)
    return img


class ImgData:
    '''Class for holding detector detector image data and metadata

    Args:
        data (numpy.ndarray): Raw detector intensity.
        header (dict): Header information.
        check_header (optional, bool): Process header to ensure that required
            parameters are present. Also sets default values for some
            parameters if not present in header. Defaults to True.
        dark_subtracted (optional, bool): Is dark current subtracted from the
            raw data? This has implications for the standard error. Defaults to
            True.
        corrs (str or list, optional): Apply corrections to image data?
            Argument is passed to the apply_corrections method. Defaults to
            "none".

    Attributes:
        data (numpy.ndarray): Intensity for each pixel.
        error (numpy.ndarray): Standard error for each pixel.
        header (dict): Header information.
    '''

    def __init__(self, data, header, check_header=True, dark_subtracted=True,
                 mask_le_dummy=True, corrs='none'):
        # process header
        if check_header:
            self.__process_header(header)
        else:
            self.header = header

        # mask pixels
        if mask_le_dummy:
            dummy_max = self.header['Dummy'] + self.header['DDummy']
            data[data <= dummy_max] = np.nan
        else:
            dummy_min = self.header['Dummy'] - self.header['DDummy']
            dummy_max = self.header['Dummy'] + self.header['DDummy']
            data[(data <= dummy_max) & (data >= dummy_min)] = np.nan

        # assign data and standard error (Poisson statistics)
        if dark_subtracted:
            self.data = np.array(data)
            self.error = np.sqrt(data + self.header['DarkConstant'])
        else:
            self.data = np.array(data) - self.header['DarkConstant']
            self.error = np.sqrt(data)

        # apply corrections (if any).
        self.apply_corrections(corrs)

    def get_coordinates(self, reference_system="normal", orientation=1):
        '''Get coordinates of detector image in various reference systems

        Possible arguments to reference system are:
        - array: Pixel centers.
        - image: Pixel centers including offset.
        - region: Pixel centers incuding offset scaled by binning size.
        - real: Real space coordinates in meters.
        - center: Pixel coordinate relative to direct beam.
        - normal: Real space coordinates relative to direct beam in meters.
        - polar: Polar coordinates. (2theta, phi).
        - wavevector: 2D wavevector coordinates. (q_x, q_z).
        - gisaxs: 2D GISAXS coordinates. (qxy, qz).

        Args:
            reference_system (str): Target reference system. Defaults to
                "normal".
            orientation (int): Desired raster orientation. Defaults to 1
                (positive horizontal as the first axis, positive horizontal as
                the second axis.

        Returns:
            (numpy.ndarray, numpy.ndarray): 2-Tuple with horizontal and
            vertical coordinates of the detector image in the desired reference
            system.
        '''
        n1, n2 = self.header["Dim_1"], self.header["Dim_2"]
        o1, o2 = self.header["Offset_1"], self.header["Offset_2"]
        b1, b2 = self.header["BSize_1"], self.header["BSize_2"]
        p1, p2 = self.header["PSize_1"], self.header["PSize_2"]
        c1, c2 = self.header["Center_1"], self.header["Center_2"]

        xx, yy = np.meshgrid(np.linspace(0.5, n1-0.5, n1),
                             np.linspace(0.5, n2-0.5, n2))

        if reference_system == "array":
            pass
        elif reference_system == "image":
            xx, yy = xx + o1, yy + o1
        elif reference_system == "region":
            xx, yy = (xx + o1) * b1, (yy + o2) * b2
        elif reference_system == "real":
            xx, yy = (xx + o1) * p1, (yy + o2) * p2
        elif reference_system == "center":
            xx, yy = xx + o1 - c1, yy + o2 - c2
        elif reference_system == "normal":
            xx, yy = (xx + o1 - c1) * p1, (yy + o2 - c2) * p2
        elif reference_system == "polar":
            xx, yy = self.__get_polar_coords()
        elif reference_system == "wavevector":
            xx, yy = self.__get_wavevector_coords(components=('qx', 'qz'))
        elif reference_system == "gisaxs":
            xx, yy = self.__get_gisaxs_coords()

        # Orient according to chosen raster (default 1)
        if orientation is not None:
            xx, yy = self.__orient_coordinates(xx, yy, orientation=orientation)

        # Positive coordinates for absolute coordinate systems
        if reference_system in ('array', 'image', 'region', 'real'):
            xx -= xx.min()
            yy -= yy.min()

        return xx, yy

    def get_q(self):
        '''Get wavevector magnitude (q) of each pixel in inverse Angstrom.

        Calculated as 4*pi/lambda*sin(theta) where lambda is the wavelength of
        the incident radiation in angstroms and theta is half the scattering
        angle (see documentation of get_tth).

        Returns:
            numpy.ndarray: Wavevector magntitude (q) of each pixel on
                the detector.
        '''
        return self.__get_wavevector_coords(components='q')

    def get_tth(self):
        '''Get scattering angle (twotheta) of each pixel in degrees

        Calculated as arccos(sd/r) where sd is the sample-detector distance and
        r is the distance from the direct beam to each pixel.

        Returns:
            numpy.ndarray: Scattering angle of each pixel on the detector.
        '''
        tth, _ = self.get_coordinates(reference_system="polar")
        return tth

    def __orient_coordinates(self, xx, yy, orientation=1):
        orientation_from = self.header['RasterOrientation']
        m1 = RASTER_MAT[orientation_from]
        m2 = RASTER_MAT[orientation]
        m = m1 @ m2

        if m[0, 0] != 0:
            if m[0, 0] == -1:
                xx = -xx
            if m[1, 1] == -1:
                yy = -yy
            return xx, yy
        else:
            if m[0, 1] == -1:
                yy = -yy
            if m[1, 0] == -1:
                xx = -xx
            return yy, xx

    def plot(self, ax=None, coords='normal', rebin=None, **kwargs):
        '''plot detector image using matplotlib

        Args:
            ax (optional): Axis to plot on
            coords (str, optional): Reference system. Defaults to 'normal'. See
                get_coordinates method for details.
            rebin (optional): Rebin the output? If a value is supplied it is
                passed to the bins argument of numpy.histogram2d. Typical usage
                is to supply an integer for the number of bins in both the
                horizontal and vertical directions.
            **kwargs: Additional keywords passed to pcolormesh.
        '''
        xx, zz = self.get_coordinates(coords)

        # polar coordinates in degrees (between 0 and 360) for plotting
        if coords == 'polar':
            xx[xx < 0] += 2*np.pi
            zz[zz < 0] += 2*np.pi
            xx, zz = xx*180/np.pi, zz*180/np.pi

        if rebin:
            # be aware that the rebinned coordinates are edge coordinates, so
            # they should only be used for plotting.
            xx, zz, data = self.__rebin2d(xx, zz, self.data, rebin)
        else:
            data = self.data

        default_kwargs = {'norm': matplotlib.colors.LogNorm()}
        if kwargs:
            kwargs = {**default_kwargs, **kwargs}
        else:
            kwargs = default_kwargs

        if ax is None:
            ax = plt.gca()

        ax.pcolormesh(xx, zz, data, **kwargs)

        if coords in AX_LABELS:
            ax.set(xlabel=AX_LABELS[coords]['x'],
                   ylabel=AX_LABELS[coords]['y'])

    def apply_corrections(self, corrs='all', **kwargs):
        '''Apply corrections to image

        The following corrections are available:

        TI: Normalization by time.
        FL_TR: Normalization by incident flux and transmitted intensity.
        SP_avg: Normalization by average solid angle of a pixel.
        SP: Solid angle correction.
        PO: Polarization correction

        Only the SP and PO corrections change the relative intensities (both
        corrections depend on the scattering angle).

        Args:
            corrs (optional): Either a string "none" or "all". "none" applies
                no corrections and "all" applies all corrections. Otherwise the
                argument should be a list of desired corrections (as str) with
                the choices being "TI", "FL_TR", "SP_avg", "SP" and "PO".
            **kwargs: Keyword arguments that can override values extracted from
                the image header. Possible arguments are: "ExposureTime",
                "TransmittedFlux" and "SourcePolarization"
        '''
        corr_function = {'TI': self.__corr_fact_TI,
                         'FL_TR': self.__corr_fact_FL_TR,
                         'SP': self.__corr_fact_SP,
                         'avg_SP': self.__corr_fact_avg_SP,
                         'PO': self.__corr_fact_PO}

        if corrs == 'all':
            corrs = corr_function.keys()
            kwargs = {'Lp': self.__get_Lp(), **kwargs}

        if corrs == 'none':
            corrs = ()

        for corr in corrs:
            if corr not in corr_function:
                raise ValueError(f'Correction {corr} not recognized')

            data_fact, error_fact = corr_function[corr](**kwargs)
            self.data *= data_fact
            self.error *= error_fact

    def __corr_fact_TI(self, **kwargs):
        if 'ExposureTime' in kwargs:
            fact = 1/kwargs['ExposureTime']
        else:
            fact = 1/self.header['ExposureTime']

        return fact, fact

    def __corr_fact_FL_TR(self, **kwargs):
        key = 'TransmittedFlux'
        if key in kwargs:
            fact = 1/kwargs[key]
        else:
            fact = 1/self.header['TransmittedFlux']

        return fact, fact

    def __corr_fact_SP(self, **kwargs):
        L0 = self.header['SampleDistance']
        if 'Lp' in kwargs:
            Lp = kwargs['Lp']
        else:
            Lp = self.__get_Lp()
        fact = (Lp/L0)**3
        return fact, fact

    def __corr_fact_avg_SP(self, **kwargs):
        L0 = self.header['SampleDistance']
        px = self.header['PSize_1']
        py = self.header['PSize_2']
        fact = L0**2 / px / py
        return fact, fact

    def __corr_fact_PO(self, **kwargs):
        L0 = self.header['SampleDistance']
        # convert polarization to fraction of horizontally polarized
        # light (rather than the PyFAI convention used in the header
        # ranging from -1 to +1).
        key = 'SourcePolarization'
        if key in kwargs:
            pol = (kwargs[key] + 1)/2
        else:
            pol = (self.header[key] + 1)/2

        if math.isclose(pol, 0.5):
            # calculation is faster for unpolarized light and we can
            # re-use the Lp calculation for the solid angle correction
            if 'Lp' in kwargs:
                Lp = kwargs['Lp']
            else:
                Lp = self.__get_Lp()

            fact = 0.5*(1 + (L0/Lp)**2)
        else:
            tth, phi = self.get_coordinates(reference_system='polar')
            sin_tth = np.sin(tth)
            # pol and (1-pol) factors are switched here compared to
            # most litterature. This is because we use the convention
            # that an azimuthal angle of 0 is horizontal on the
            # detector.
            fact = ((1-pol)*(1-(np.sin(phi)*sin_tth)**2) +
                    pol*(1-(np.cos(phi)*sin_tth)**2))

        # we compensate for the reduction in intensity due to
        # polarization, so divide by the computed factor
        return 1/fact, 1/fact

    def __get_Lp(self):
        xx, zz = self.get_coordinates(reference_system='normal')
        yy = self.header['SampleDistance']
        return np.sqrt(xx**2 + yy**2 + zz**2)

    def __rebin2d(self, xx, yy, data, bins):
        img_rb, xx_rb, yy_rb = np.histogram2d(x=xx.flatten(),
                                              y=yy.flatten(),
                                              weights=data.flatten(),
                                              bins=bins)
        return xx_rb, yy_rb, img_rb.T

    def __get_polar_coords(self):
        lab_xx, lab_zz = self.get_coordinates('normal')
        lab_yy = self.header['SampleDistance']
        r = np.sqrt(lab_xx**2 + lab_yy**2 + lab_zz**2)
        tth = np.arccos(lab_yy/r)
        phi = np.arctan2(lab_zz, lab_xx)

        # We use the convention that an azimuthal angle of zero should
        # be horizontal on the detector. For asymmetric raster
        # orientations, we thus need to add 90 degrees to phi as the
        # horizontal and vertical axes are flipped.
        if self.header['RasterOrientation'] > 4:
            phi += np.pi/2

        # reset phi to be between 0 and 2*pi.
        phi[phi < 0] += 2*np.pi

        return tth, phi

    def __get_wavevector_coords(self, components=None):
        if components is None:
            components = ('qx', 'qz')

        if isinstance(components, str):
            components = (components)

        tth, phi = self.get_coordinates('polar')

        unit = 1e-10  # angstrom
        theta = tth/2
        sin_theta = np.sin(theta)
        q = 4*np.pi/self.header['WaveLength']*sin_theta*unit

        # if we just need the magnitude, return immidiately
        if len(components) == 1 and components[0] == 'q':
            return q

        # otherwise, compute q-components
        cos_theta = np.cos(theta)
        qx = q*cos_theta*np.cos(phi)
        qy = q*sin_theta
        qz = q*cos_theta*np.sin(phi)

        component_dict = {'qx': qx, 'qy': qy, 'qz': qz, 'q': q}

        if len(components) == 1:
            return component_dict[components[0]]
        else:
            result = []
            for component in components:
                result.append(component_dict[component])
            return tuple(result)

    def __get_gisaxs_coords(self):
        qx, qy, qz = self.__get_wavevector_coords(('qx', 'qy', 'qz'))

        alpha_i = self.header['IncidentAngle'] * np.pi / 180
        sin_alpha_i = np.sin(alpha_i)
        cos_alpha_i = np.cos(alpha_i)

        # a rotation of the sample stage by alpha_i corresponds to a rotation
        # of (qy, qz) coordinates around qx by alpha_i
        qy_r = qy * cos_alpha_i - qz * sin_alpha_i
        qz_r = qy * sin_alpha_i + qz * cos_alpha_i
        signed_qxy = np.sqrt(qx**2 + qy_r**2)*np.sign(qx)
        return signed_qxy, qz_r

    def __process_header(self, header):
        if not isinstance(header, dict):
            raise TypeError("Input header must be a dictionary")

        self.header = header
        missing_required_keys = []
        for key, (dtype, dval, cat) in KEYWORDS.items():
            if key in header and isinstance(header[key], dtype):
                self.header[key] = header[key]
            elif key in header:
                self.header[key] = dtype(header[key])
            elif dval is not None:
                self.header[key] = dval
            elif cat in ('optional', 'intensity'):
                self.header[key] = None
            else:
                missing_required_keys.append(key)

            if len(missing_required_keys) > 0:
                raise KeyError("Required scattering geometry keywords {} not"
                               "found in header".format(missing_required_keys))
