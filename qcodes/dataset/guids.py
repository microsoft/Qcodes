from typing import Union, Dict
import time

import numpy as np

from qcodes.config import Config


def generate_guid(timeint: Union[int, None]=None) -> str:
    """
    Generate a guid string to go into the GUID column of the runs table.
    The GUID is based on the GUID-components in the qcodesrc file.
    The generated string is of the format
    '12345678-1234-1234-1234-123456789abc', where the first eight hex numbers
    comprise the 4 byte sample code, the next 2 hex numbers comprise the 1 byte
    location, the next 2+4 hex numbers are the 3 byte work station code, and
    the final 4+12 hex number give the 8 byte integer time in ms since epoch
    time

    Args:
        timeint: An integer of miliseconds since unix epoch time
    """
    cfg = Config()

    try:
        guid_comp = cfg['GUID_components']
    except KeyError:
        raise RuntimeError('Invalid QCoDeS config file! No GUID_components '
                           'specified. Can not proceed.')

    location = guid_comp['location']
    station = guid_comp['work_station']
    sample = guid_comp['sample']

    if timeint is None:
        # ms resolution, checked on windows
        timeint = int(np.round(time.time()*1000))

    loc_str = f'{location:02x}'
    stat_str = f'{station:06x}'
    smpl_str = f'{sample:08x}'
    time_str = f'{timeint:016x}'

    guid = (f'{smpl_str}-{loc_str}{stat_str[:2]}-{stat_str[2:]}-'
            f'{time_str[:4]}-{time_str[4:]}')

    return guid


def parse_guid(guid: str) -> Dict[str, int]:
    """
    Parse a guid back to its four constituents

    Args:
        guid: a valid guid str

    Returns:
        A dict with keys 'location', 'work_station', 'sample', and 'time'
          and integer values
    """
    guid = guid.replace('-', '')
    components = {}
    components['sample'] = int(guid[:8], base=16)
    components['location'] = int(guid[8:10], base=16)
    components['work_station'] = int(guid[10:16], base=16)
    components['time'] = int(guid[16:], base=16)

    return components
