from typing import Union, Dict, Sequence, List, Optional

import time
import re


import numpy as np
import qcodes as qc


_guid_pattern = re.compile(r'^[\da-f]{8}-([\da-f]{4}-){3}[\da-f]{12}$')


def generate_guid(timeint: Union[int, None]=None,
                  sampleint: Union[int, None]=None) -> str:
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
        sampleint: A code for the sample
    """
    cfg = qc.config

    try:
        guid_comp = cfg['GUID_components']
    except KeyError:
        raise RuntimeError('Invalid QCoDeS config file! No GUID_components '
                           'specified. Can not proceed.')

    location = guid_comp['location']
    station = guid_comp['work_station']

    if timeint is None:
        # ms resolution, checked on windows
        timeint = int(np.round(time.time()*1000))
    if sampleint is None:
        sampleint = guid_comp['sample']

    if sampleint == 0:
        sampleint = int('a'*8, base=16)

    loc_str = f'{location:02x}'
    stat_str = f'{station:06x}'
    smpl_str = f'{sampleint:08x}'
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
          as integer values
    """
    guid = guid.replace('-', '')
    components = {}
    components['sample'] = int(guid[:8], base=16)
    components['location'] = int(guid[8:10], base=16)
    components['work_station'] = int(guid[10:16], base=16)
    components['time'] = int(guid[16:], base=16)

    return components


def set_guid_location_code() -> None:
    """
    Interactive function to set the location code.
    """
    cfg = qc.config
    old_loc = cfg['GUID_components']['location']
    print(f'Updating GUID location code. Current location code is: {old_loc}')
    if old_loc != 0:
        print('That is a non-default location code. Perhaps you should not '
              'change it? Re-enter that code to leave it unchanged.')
    loc_str = input('Please enter the new location code (1-256): ')
    try:
        location = int(loc_str)
    except ValueError:
        print('The location code must be an integer. No update performed.')
        return
    if not(257 > location > 0):
        print('The location code must be between 1 and 256 (both included). '
              'No update performed')
        return

    cfg['GUID_components']['location'] = location
    cfg.save_to_home()


def set_guid_work_station_code() -> None:
    """
    Interactive function to set the work station code
    """
    cfg = qc.config
    old_ws = cfg['GUID_components']['work_station']
    print('Updating GUID work station code. '
          f'Current work station code is: {old_ws}')
    if old_ws != 0:
        print('That is a non-default work station code. Perhaps you should not'
              ' change it? Re-enter that code to leave it unchanged.')
    ws_str = input('Please enter the new work station code (1-16777216): ')
    try:
        work_station = int(ws_str)
    except ValueError:
        print('The work station code must be an integer. No update performed.')
        return
    if not(16777216 > work_station > 0):
        print('The work staion code must be between 1 and 256 (both included).'
              ' No update performed')
        return

    cfg['GUID_components']['work_station'] = work_station
    cfg.save_to_home()


def filter_guids_by_parts(guids: Sequence[str],
                          location: Optional[int] = None,
                          sample_id: Optional[int] = None,
                          work_station: Optional[int] = None) -> List[str]:
    """
    Filter a sequence of GUIDs by location, sample_id and/or work_station.

    Args:
        guids: Sequence of guids that should be filtered.
        location: Location code to match
        sample_id: Sample_id to match
        work_station: Workstation to match

    Returns:
        A list of GUIDs that matches the supplied parts.
    """
    matched_guids = []
    for guid in guids:
        guid_dict = parse_guid(guid)
        match = True
        if sample_id is not None:
            if guid_dict['sample'] != sample_id:
                match = False
        if location is not None:
            if guid_dict['location'] != location:
                match = False
        if work_station is not None:
            if guid_dict['work_station'] != work_station:
                match = False

        if match:
            matched_guids.append(guid)
    return matched_guids


def validate_guid_format(guid: str) -> None:
    """
    Validate the format of the given guid. This function does not check the
    correctness of the data inside the guid (e.g. timestamps in the far
    future)
    """
    if _guid_pattern.match(guid):
            return
    else:
        raise ValueError(f'Did not receive a valid guid. Got {guid}')
