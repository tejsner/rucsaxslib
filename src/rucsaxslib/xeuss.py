'''
Module to deal with reading data from the Xeuss3 instrument.
'''
import numpy as np
import re
import json
import fabio


def read_dat_bio(fn):
    '''Read BioCube .dat files

    Biocube dat files contains two types of metadata:

    1) image metadata inherited from the header of files uses for reduction
    2) metadata about the analysis performed (average, subtraction, ...)

    The image metadata is saved as-is and the analysis metadata is then added
    using the key _BIO_ANALYSIS.

    The special key _COL_NAMES, containing the names of the data columns,
    is also added to the metadata

    Args:
        fn (str): Filename

    Returns:
        (dict, numpy.ndarray): Metadata, data
    '''

    # open file, split lines into header and data
    with open(fn) as fs:
        section_being_parsed = None
        header_lines = []
        data_lines = []

        line = fs.readline()
        while line:
            if line.startswith('### HEADER'):
                section_being_parsed = 'header'
            elif line.startswith('### DATA'):
                section_being_parsed = 'data'

            if section_being_parsed == 'header':
                header_lines.append(line)
            elif section_being_parsed == 'data':
                data_lines.append(line)

            line = fs.readline()

    # parse header, save as dict through json parser
    # split header into two dictionaries
    #   bio_header: information related to biocube reduction
    #   header: normal header information
    header_str = ''
    for line in header_lines[1:]:
        header_str += line.strip('#').strip()
    bio_header = json.loads(header_str)
    header = bio_header.pop('imageHeader')

    # parse data
    col_names = data_lines[3].split()[1:]
    row_data = []
    for line in data_lines[4:]:
        row = line.split()
        if len(row) == len(col_names):
            row_data.append(row)

    data = []
    for i, _ in enumerate(col_names):
        data.append([float(x[i]) for x in row_data])

    # add custom fields to the header
    header['_BIO_ANALYSIS'] = bio_header
    header['_COL_NAMES'] = col_names

    return (header, np.array(data).T)


def read_dat(fn):
    '''Read .dat files create by Xeuss3

    The special key _COL_NAMES, containing the names of the data columns,
    is added to the metadata.

    Args:
        fn (str): Filename

    Returns:
        (dict, numpy.ndarray): Metadata, data
    '''
    # open file, split lines into header (starting with #) and data (rest)
    with open(fn) as fs:
        header_lines = []
        data_lines = []
        line = fs.readline()
        while line:
            if line.startswith('#'):
                header_lines.append(line)
            else:
                data_lines.append(line)

            line = fs.readline()

    # parse header, save as dict
    header = {}
    for line in header_lines:
        r = re.search(r'^# (\S+)\s+(.*)$', line)
        if r is not None:
            header[r[1]] = r[2]

    # parse data, save as dict with column names as key
    col_names = data_lines[0].split()
    row_data = []
    for line in data_lines[1:]:
        row = line.split()
        if len(row) == len(col_names):
            row_data.append(row)

    data = []
    for i, _ in enumerate(col_names):
        data.append([float(x[i]) for x in row_data])

    # add custom fields to the header
    header['_COL_NAMES'] = col_names

    return (header, np.array(data).T)


def read_edf(fn):
    '''
    read image files (.edf) created by Xeuss3.

    Uses the fabio library.

    Image is returned as a numpy.ndarray with indices corresponding to the
    indices of the detector (starting from the top-left when looking at the
    detector).

    Values are given as photon counts. Masked pixels take the value numpy.nan

    Args:
        fn (str): Filename

    Returns:
        (dict, numpy.ndarray): Metadata, data
    '''
    img = fabio.open(fn)
    header, data = dict(img.header), img.data
    data[data <= -1] = np.nan
    return header, data


def read_spec(fn):
    '''
    Get all scans from a SPEC logfile

    Args:
        fn (str): Filename

    Returns:
        dict: SPEC logfile scans. Key is scan id and value is (metadata, data)
            for the given scan
    '''
    with open(fn) as fs:
        scans = {}
        id = None
        line = fs.readline()
        while line:
            if line.startswith("#S "):
                id = int(line.split()[1])
                scans[id] = []
            if id is not None and not line.startswith("#C "):
                scans[id].append(line.strip())

            line = fs.readline()

    data = {}
    for id in scans:
        header = []
        previous_line = '#'
        for line in scans[id]:
            if not line.startswith('#') and previous_line.startswith('#'):
                colnames = previous_line.strip('#L ').split('  ')
                break

            header.append(line)
            previous_line = line
        data[id] = (header, np.genfromtxt(scans[id], names=colnames))

    return data
