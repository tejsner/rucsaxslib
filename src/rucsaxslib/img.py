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
            'ExposureTime': (float, None, 'intensity'),
            'DarkConstant': (float, 0.0, 'intensity'),
            'TransmittedFlux': (float, None, 'intensity'),
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
    if engine == 'fabio':
        faimg = fabio.open(filename)

        for key in header_rename:
            faimg.header[header_rename[key]] = faimg.header[key]

        for key in header_extra:
            faimg.header[key] = header_extra[key]

        return ImgData(faimg.data, faimg.header, **kwargs)
    else:
        raise NotImplementedError(f'Engine {engine} not implemented')


def from_rucsaxs(filename, **kwargs):
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
    def __init__(self, data, header, check_header=True, dark_subtracted=True,
                 mask_le_dummy=True, apply_corrs=False):
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

        # apply corrections?
        if apply_corrs and isinstance(apply_corrs, bool):
            Lp = self.__get_Lp()
            self.apply_corrections(Lp=Lp)
        elif apply_corrs and isinstance(apply_corrs, list):
            if 'PO' in apply_corrs and 'SP' in apply_corrs:
                Lp = self.__get_Lp()
            self.apply_corrections(corrs=apply_corrs, Lp=Lp)

    def get_coordinates(self, reference_system="normal", orientation=1,
                        wavevector_components=None):
        n1, n2 = self.header["Dim_1"], self.header["Dim_2"]
        o1, o2 = self.header["Offset_1"], self.header["Offset_2"]
        b1, b2 = self.header["BSize_1"], self.header["BSize_2"]
        p1, p2 = self.header["PSize_1"], self.header["PSize_2"]
        c1, c2 = self.header["Center_1"], self.header["Center_2"]

        xx, yy = np.meshgrid(np.linspace(0.5, n1-0.5, n1),
                             np.linspace(0.5, n2-0.5, n2))

        if reference_system == "array":
            return xx, yy
        elif reference_system == "image":
            return xx + o1, yy + o1
        elif reference_system == "region":
            return (xx + o1) * b1, (yy + o2) * b2
        elif reference_system == "real":
            return (xx + o1) * p1, (yy + o2) * p2
        elif reference_system == "center":
            return xx + o1 - c1, yy + o2 - c2
        elif reference_system == "normal":
            return (xx + o1 - c1) * p1, (yy + o2 - c2) * p2
        elif reference_system == "polar":
            return self.__get_polar_coords()
        elif reference_system == "wavevector":
            return self.__get_wavevector_coords(wavevector_components)
        elif reference_system == "gisaxs":
            return self.__get_gisaxs_coords()

    def get_q(self):
        return self.get_coordinates(reference_system="wavevector",
                                    wavevector_components='q')

    def orient_coordinates(self, xx, yy, orientation=1):
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
        xx, zz = self.get_coordinates(coords)
        xx, zz = self.orient_coordinates(xx, zz, orientation=1)

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
        corr_function = {'TI': self.__corr_fact_TI,
                         'FL_TR': self.__corr_fact_FL_TR,
                         'SP': self.__corr_fact_SP,
                         'avg_SP': self.__corr_fact_avg_SP,
                         'PO': self.__corr_fact_PO}

        if corrs == 'all':
            corrs = corr_function.keys()

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
