# awg file -> (what, we, put, into, make_send_and_load_awg_file)
# This module parses an awg file using THREE sub-parser. This code could
# probably be streamlined somewhat.

import struct

import numpy as np

AWG_FILE_FORMAT = {
    'MAGIC': 'h',
    'VERSION': 'h',
    'SAMPLING_RATE': 'd',    # d
    'REPETITION_RATE': 'd',    # # NAME?
    'HOLD_REPETITION_RATE': 'h',    # True | False
    'CLOCK_SOURCE': 'h',    # Internal | External
    'REFERENCE_SOURCE': 'h',    # Internal | External
    'EXTERNAL_REFERENCE_TYPE': 'h',    # Fixed | Variable
    'REFERENCE_CLOCK_FREQUENCY_SELECTION': 'h',  # 10 MHz | 20 MHz | 100 MHz
    'REFERENCE_MULTIPLIER_RATE': 'h',    #
    'DIVIDER_RATE': 'h',   # 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256
    'TRIGGER_SOURCE': 'h',    # Internal | External
    'INTERNAL_TRIGGER_RATE': 'd',    #
    'TRIGGER_INPUT_IMPEDANCE': 'h',    # 50 ohm | 1 kohm
    'TRIGGER_INPUT_SLOPE': 'h',    # Positive | Negative
    'TRIGGER_INPUT_POLARITY': 'h',    # Positive | Negative
    'TRIGGER_INPUT_THRESHOLD': 'd',    #
    'EVENT_INPUT_IMPEDANCE': 'h',    # 50 ohm | 1 kohm
    'EVENT_INPUT_POLARITY': 'h',    # Positive | Negative
    'EVENT_INPUT_THRESHOLD': 'd',
    'JUMP_TIMING': 'h',    # Sync | Async
    'INTERLEAVE': 'h',    # On |  This setting is stronger than coupling
    'ZEROING': 'h',    # On | Off
    'COUPLING': 'h',    # The Off | Pair | All setting is weaker than .
    'RUN_MODE': 'h',    # Continuous | Triggered | Gated | Sequence
    'WAIT_VALUE': 'h',    # First | Last
    'RUN_STATE': 'h',    # On | Off
    'INTERLEAVE_ADJ_PHASE': 'd',
    'INTERLEAVE_ADJ_AMPLITUDE': 'd',
    'EVENT_JUMP_MODE': 'h',  # Event jump | Dynamic jump
    'TABLE_JUMP_STROBE': 'h',  # On
    'TABLE_JUMP_DEFINITION': 'ignore',  # Array of tablejump
    # ---------------------
    # Channel 1 settings
    # ---------------------
    'OUTPUT_WAVEFORM_NAME_1': 's',  # if in continuous mode
    'DAC_RESOLUTION_1': 'h',  # 8 | 10
    'CHANNEL_STATE_1': 'h',  # On | Off
    'ANALOG_DIRECT_OUTPUT_1': 'h',  # On | Off
    'ANALOG_FILTER_1': 'h',  # Enum type.
    'ANALOG_METHOD_1': 'h',  # Amplitude/Offset, High/Low
    # When the Input Method is High/Low, it is skipped.
    'ANALOG_AMPLITUDE_1': 'd',
    # When the Input Method is High/Low, it is skipped.
    'ANALOG_OFFSET_1': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'ANALOG_HIGH_1': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'ANALOG_LOW_1': 'd',
    'MARKER1_SKEW_1': 'd',
    'MARKER1_METHOD_1': 'h',  # Amplitude/Offset, High/Low
    # When the Input Method is High/Low, it is skipped.
    'MARKER1_AMPLITUDE_1': 'd',
    # When the Input Method is High/Low, it is skipped.
    'MARKER1_OFFSET_1': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'MARKER1_HIGH_1': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'MARKER1_LOW_1': 'd',
    'MARKER2_SKEW_1': 'd',
    'MARKER2_METHOD_1': 'h',  # Amplitude/Offset, High/Low
    # When the Input Method is High/Low, it is skipped.
    'MARKER2_AMPLITUDE_1': 'd',
    # When the Input Method is High/Low, it is skipped.
    'MARKER2_OFFSET_1': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'MARKER2_HIGH_1': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'MARKER2_LOW_1': 'd',
    'DIGITAL_METHOD_1': 'h',  # Amplitude/Offset, High/Low
    # When the Input Method is High/Low, it is skipped.
    'DIGITAL_AMPLITUDE_1': 'd',
    # When the Input Method is High/Low, it is skipped.
    'DIGITAL_OFFSET_1': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'DIGITAL_HIGH_1': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'DIGITAL_LOW_1': 'd',
    'EXTERNAL_ADD_1': 'h',  # AWG5000 only
    'PHASE_DELAY_INPUT_METHOD_1':   'h',  # Phase/DelayInme/DelayInints
    'PHASE_1': 'd',  # When the Input Method is not Phase, it is skipped.
    # When the Input Method is not DelayInTime, it is skipped.
    'DELAY_IN_TIME_1': 'd',
    # When the Input Method is not DelayInPoint, it is skipped.
    'DELAY_IN_POINTS_1': 'd',
    'CHANNEL_SKEW_1': 'd',
    'DC_OUTPUT_LEVEL_1': 'd',  # V
    # ---------------------
    # ---------------------
    'OUTPUT_WAVEFORM_NAME_2': 's',
    'DAC_RESOLUTION_2': 'h',  # 8 | 10
    'CHANNEL_STATE_2': 'h',  # On | Off
    'ANALOG_DIRECT_OUTPUT_2': 'h',  # On | Off
    'ANALOG_FILTER_2': 'h',  # Enum type.
    'ANALOG_METHOD_2': 'h',  # Amplitude/Offset, High/Low
    # When the Input Method is High/Low, it is skipped.
    'ANALOG_AMPLITUDE_2': 'd',
    # When the Input Method is High/Low, it is skipped.
    'ANALOG_OFFSET_2': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'ANALOG_HIGH_2': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'ANALOG_LOW_2': 'd',
    'MARKER1_SKEW_2': 'd',
    'MARKER1_METHOD_2': 'h',  # Amplitude/Offset, High/Low
    # When the Input Method is High/Low, it is skipped.
    'MARKER1_AMPLITUDE_2': 'd',
    # When the Input Method is High/Low, it is skipped.
    'MARKER1_OFFSET_2': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'MARKER1_HIGH_2': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'MARKER1_LOW_2': 'd',
    'MARKER2_SKEW_2': 'd',
    'MARKER2_METHOD_2': 'h',  # Amplitude/Offset, High/Low
    # When the Input Method is High/Low, it is skipped.
    'MARKER2_AMPLITUDE_2': 'd',
    # When the Input Method is High/Low, it is skipped.
    'MARKER2_OFFSET_2': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'MARKER2_HIGH_2': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'MARKER2_LOW_2': 'd',
    'DIGITAL_METHOD_2': 'h',  # Amplitude/Offset, High/Low
    # When the Input Method is High/Low, it is skipped.
    'DIGITAL_AMPLITUDE_2': 'd',
    # When the Input Method is High/Low, it is skipped.
    'DIGITAL_OFFSET_2': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'DIGITAL_HIGH_2': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'DIGITAL_LOW_2': 'd',
    'EXTERNAL_ADD_2': 'h',  # AWG5000 only
    'PHASE_DELAY_INPUT_METHOD_2':   'h',  # Phase/DelayInme/DelayInints
    'PHASE_2': 'd',  # When the Input Method is not Phase, it is skipped.
    # When the Input Method is not DelayInTime, it is skipped.
    'DELAY_IN_TIME_2': 'd',
    # When the Input Method is not DelayInPoint, it is skipped.
    'DELAY_IN_POINTS_2': 'd',
    'CHANNEL_SKEW_2': 'd',
    'DC_OUTPUT_LEVEL_2': 'd',  # V
    # ---------------------
    # ---------------------
    'OUTPUT_WAVEFORM_NAME_3': 's',
    'DAC_RESOLUTION_3': 'h',  # 8 | 10
    'CHANNEL_STATE_3': 'h',  # On | Off
    'ANALOG_DIRECT_OUTPUT_3': 'h',  # On | Off
    'ANALOG_FILTER_3': 'h',  # Enum type.
    'ANALOG_METHOD_3': 'h',  # Amplitude/Offset, High/Low
    # When the Input Method is High/Low, it is skipped.
    'ANALOG_AMPLITUDE_3': 'd',
    # When the Input Method is High/Low, it is skipped.
    'ANALOG_OFFSET_3': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'ANALOG_HIGH_3': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'ANALOG_LOW_3': 'd',
    'MARKER1_SKEW_3': 'd',
    'MARKER1_METHOD_3': 'h',  # Amplitude/Offset, High/Low
    # When the Input Method is High/Low, it is skipped.
    'MARKER1_AMPLITUDE_3': 'd',
    # When the Input Method is High/Low, it is skipped.
    'MARKER1_OFFSET_3': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'MARKER1_HIGH_3': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'MARKER1_LOW_3': 'd',
    'MARKER2_SKEW_3': 'd',
    'MARKER2_METHOD_3': 'h',  # Amplitude/Offset, High/Low
    # When the Input Method is High/Low, it is skipped.
    'MARKER2_AMPLITUDE_3': 'd',
    # When the Input Method is High/Low, it is skipped.
    'MARKER2_OFFSET_3': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'MARKER2_HIGH_3': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'MARKER2_LOW_3': 'd',
    'DIGITAL_METHOD_3': 'h',  # Amplitude/Offset, High/Low
    # When the Input Method is High/Low, it is skipped.
    'DIGITAL_AMPLITUDE_3': 'd',
    # When the Input Method is High/Low, it is skipped.
    'DIGITAL_OFFSET_3': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'DIGITAL_HIGH_3': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'DIGITAL_LOW_3': 'd',
    'EXTERNAL_ADD_3': 'h',  # AWG5000 only
    'PHASE_DELAY_INPUT_METHOD_3':   'h',  # Phase/DelayInme/DelayInints
    'PHASE_3': 'd',  # When the Input Method is not Phase, it is skipped.
    # When the Input Method is not DelayInTime, it is skipped.
    'DELAY_IN_TIME_3': 'd',
    # When the Input Method is not DelayInPoint, it is skipped.
    'DELAY_IN_POINTS_3': 'd',
    'CHANNEL_SKEW_3': 'd',
    'DC_OUTPUT_LEVEL_3': 'd',  # V
    # ---------------------
    # ---------------------
    'OUTPUT_WAVEFORM_NAME_4': 's',
    'DAC_RESOLUTION_4': 'h',  # 8 | 10
    'CHANNEL_STATE_4': 'h',  # On | Off
    'ANALOG_DIRECT_OUTPUT_4': 'h',  # On | Off
    'ANALOG_FILTER_4': 'h',  # Enum type.
    'ANALOG_METHOD_4': 'h',  # Amplitude/Offset, High/Low
    # When the Input Method is High/Low, it is skipped.
    'ANALOG_AMPLITUDE_4': 'd',
    # When the Input Method is High/Low, it is skipped.
    'ANALOG_OFFSET_4': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'ANALOG_HIGH_4': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'ANALOG_LOW_4': 'd',
    'MARKER1_SKEW_4': 'd',
    'MARKER1_METHOD_4': 'h',  # Amplitude/Offset, High/Low
    # When the Input Method is High/Low, it is skipped.
    'MARKER1_AMPLITUDE_4': 'd',
    # When the Input Method is High/Low, it is skipped.
    'MARKER1_OFFSET_4': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'MARKER1_HIGH_4': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'MARKER1_LOW_4': 'd',
    'MARKER2_SKEW_4': 'd',
    'MARKER2_METHOD_4': 'h',  # Amplitude/Offset, High/Low
    # When the Input Method is High/Low, it is skipped.
    'MARKER2_AMPLITUDE_4': 'd',
    # When the Input Method is High/Low, it is skipped.
    'MARKER2_OFFSET_4': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'MARKER2_HIGH_4': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'MARKER2_LOW_4': 'd',
    'DIGITAL_METHOD_4': 'h',  # Amplitude/Offset, High/Low
    # When the Input Method is High/Low, it is skipped.
    'DIGITAL_AMPLITUDE_4': 'd',
    # When the Input Method is High/Low, it is skipped.
    'DIGITAL_OFFSET_4': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'DIGITAL_HIGH_4': 'd',
    # When the Input Method is Amplitude/Offset, it is skipped.
    'DIGITAL_LOW_4': 'd',
    'EXTERNAL_ADD_4': 'h',  # AWG5000 only
    'PHASE_DELAY_INPUT_METHOD_4':   'h',  # Phase/DelayInme/DelayInints
    'PHASE_4': 'd',  # When the Input Method is not Phase, it is skipped.
    # When the Input Method is not DelayInTime, it is skipped.
    'DELAY_IN_TIME_4': 'd',
    # When the Input Method is not DelayInPoint, it is skipped.
    'DELAY_IN_POINTS_4': 'd',
    'CHANNEL_SKEW_4': 'd',
    'DC_OUTPUT_LEVEL_4': 'd',  # V
    }

# Note: this dictionary is dynamically updated by _parser1
AWG_FILE_FORMAT_WAV = {
    'WAVEFORM_NAME': 's',
    'WAVEFORM_TYPE': 'h',
    'WAVEFORM_LENGTH': 'l',
    'WAVEFORM_TIMESTAMP': '8H',
    'WAVEFORM_DATA_4': None
    }

AWG_FILE_FORMAT_SEQ = {
    'SEQUENCE_WAIT': 'h',  # On | Off
    'SEQUENCE_LOOP': 'l',  # 0=infinite
    'SEQUENCE_JUMP': 'h',  # OFF:0, INDEX: element #, NEXT: -1
    'SEQUENCE_GOTO': 'h',  # 0 if GOTO is OFF, else an element number
    'SEQUENCE_WAVEFORM': 's',
    'SEQUENCE_IS': '2h',  # Yes | No
    'SEQUENCE_SUBSEQ': 's'
    }

AWG_TRANSLATER = {
    'HOLD_REPETITION_RATE': {0: 'False', 1: 'True'},
    'CLOCK_SOURCE': {1: 'Internal', 2: 'External'},
    'REFERENCE_SOURCE': {1: 'Internal', 2: 'External'},
    'EXTERNAL_REFERENCE_TYPE': {1: 'Fixed', 2: 'Variable'},
    'TRIGGER_SOURCE': {1: 'Internal', 2: 'External'},
    'TRIGGER_INPUT_IMPEDANCE': {1: '50 Ohm', 2: '1 kOhm'},
    'TRIGGER_INPUT_SLOPE': {1: 'Positive', 2: 'Negative'},
    'TRIGGER_INPUT_POLARITY': {1: 'Positive', 2: 'Negative'},
    'EVENT_INPUT_IMPEDANCE': {1: '50 Ohm', 2: '1 kOhm'},
    'EVENT_INPUT_SLOPE': {1: 'Positive', 2: 'Negative'},
    'EVENT_INPUT_POLARITY': {1: 'Positive', 2: 'Negative'},
    'JUMP_TIMING': {1: 'Sync', 2: 'Async'},
    'RUN_MODE': {1: 'Continuous', 2: 'Triggered', 3: 'Gated', 4: 'Sequence'},
    'WAIT_VALUE': {1: 'First', 2: 'Last'}
    }


def _unpacker(binaryarray, dacbitdepth=14):
    """
    Unpacks an awg-file integer wave into a waveform and two markers
    in the same way as the AWG does. This can be useful for checking
    how the signals are going to be interpreted by the instrument.

    Args:
        binaryarray (numpy.ndarray): A numpy array containing the
            packed waveform and markers.
        dacbitdepth (int): Specifies the bit depth for the digitisation
        of the waveform. Allowed values: 14, 8. Default: 14.

    Returns:
        tuple (numpy.ndarray, numpy.ndarray, numpy.ndarray): The waveform
            scaled to have values from -1 to 1, marker 1, marker 2.
    """

    wflength = len(binaryarray)
    wf = np.zeros(wflength)
    m1 = np.zeros(wflength)
    m2 = np.zeros(wflength)

    for ii, bitnum in enumerate(binaryarray):
        bitstring = bin(bitnum)[2:].zfill(16)
        m2[ii] = int(bitstring[0])
        m1[ii] = int(bitstring[1])
        wf[ii] = (int(bitstring[2:], base=2)-2**13)/2**13
        # print(bitstring, int(bitstring[2:], base=2))

    return wf, m1, m2


def _unwrap(bites, fmt):
    """
    Helper function for interpreting the bytes from the awg file.

    Args:
        bites (bytes): a bytes object
        fmt (str): the format string (either 's', 'h' or 'd')

    Returns:
        Union[str, int, tuple]
    """

    if fmt == 's':
        value = bites[:-1].decode('ascii')
    elif fmt == 'ignore':
        value = 'Not read'
    else:
        value = struct.unpack('<'+fmt, bites)
        if len(value) == 1:
            value = value[0]

    return value


def _getendingnumber(string):
    """
    Helper function to extract the last number of a string

    Args:
        string (str): A .awg field name, like SEQUENCE_JUMP_23

    Returns
        (int, str): The number and the shortened string,
          e.g. 'SEQUENCE_JUMP_23' -> (23, 'SEQUENCE_JUMP_')
    """

    num = ''

    for char in string[::-1]:
        if char.isdigit():
            num += char
        else:
            break

    return (int(num[::-1]), string[:-len(num)])


awgfilepath = ('/Users/william/AuxiliaryQCoDeS/AWGhelpers/awgfiles/' +
               'customawgfile.awg')

awgfilepath2 = ('/Users/william/AuxiliaryQCoDeS/AWGhelpers/awgfiles/' +
                'machinemadefortest.awg')


def _parser1(awgfilepath):
    """
    Helper function doing the heavy lifting of reading and understanding the
    binary .awg file format.

    Args:
        awgfilepath (str): The absolute path of the awg file to read

    Returns:
        (dict, list, list): Instrument settings, waveforms, sequencer settings
    """

    instdict = {}
    waveformlist = [[], []]
    sequencelist = [[], []]

    with open(awgfilepath, 'rb') as fid:

        while True:
            chunk = fid.read(8)
            if not chunk:
                break

            (namelen, valuelen) = struct.unpack('<II', chunk)

            rawname = fid.read(namelen)
            rawvalue = fid.read(valuelen)

            name = rawname[:-1].decode('ascii')  # remove NULL termination char

            if name.startswith('WAVEFORM'):

                namestop = name[name.find('_')+1:].find('_')+name.find('_')
                lookupname = name[:namestop+1]

                if 'DATA' in name:
                    fmtstr = '{}H'.format(wfmlen)
                    AWG_FILE_FORMAT_WAV['WAVEFORM_DATA'] = fmtstr

                value = _unwrap(rawvalue, AWG_FILE_FORMAT_WAV[lookupname])
                (number, barename) = _getendingnumber(name)
                fieldname = barename + '{}'.format(number-20)
                waveformlist[0].append(fieldname)
                waveformlist[1].append(value)

                if 'LENGTH' in name:
                    wfmlen = value

                continue

            if name.startswith('SEQUENCE'):

                namestop = name[name.find('_')+1:].find('_')+name.find('_')
                lookupname = name[:namestop+1]
                value = _unwrap(rawvalue, AWG_FILE_FORMAT_SEQ[lookupname])
                sequencelist[0].append(name)
                sequencelist[1].append(value)

                continue

            else:
                value = _unwrap(rawvalue, AWG_FILE_FORMAT[name])

            if name in AWG_TRANSLATER:
                value = AWG_TRANSLATER[name][value]

            instdict.update({name: value})

    return instdict, waveformlist, sequencelist


def _parser2(waveformlist):
    """
    Cast the waveformlist from _parser1 into a dict used by _parser3.

    Args:
        waveformlist (list[list, list]): A list of lists of waveforms from
          _parser1

    Returns:
        dict: A dictionary with keys waveform name and values for marker1,
            marker2, and the waveform as np.arrays
    """

    outdict = {}

    for (fieldname, fieldvalue) in zip(waveformlist[0], waveformlist[1]):
        if 'NAME' in fieldname:
            name = fieldvalue
        if 'DATA' in fieldname:
            value = _unpacker(fieldvalue)
            outdict.update({name: {'m1': value[1], 'm2': value[2],
                                   'wfm': value[0]}})

    return outdict


def _parser3(sequencelist, wfmdict):
    """
    The final parser! OMG+1
    """

    sequencedict = {'SEQUENCE_WAIT': [],
                    'SEQUENCE_LOOP': [],
                    'SEQUENCE_JUMP': [],
                    'SEQUENCE_GOTO': [],
                    'SEQUENCE_WAVEFORM_NAME_CH_1': [],
                    'SEQUENCE_WAVEFORM_NAME_CH_2': [],
                    'SEQUENCE_WAVEFORM_NAME_CH_3': [],
                    'SEQUENCE_WAVEFORM_NAME_CH_4': []
    }

    for fieldname, fieldvalue in zip(sequencelist[0], sequencelist[1]):

        seqnum, name = _getendingnumber(fieldname)

        if 'WAVEFORM' not in name:
            sequencedict[name[:-1]].append(fieldvalue)
        else:
            sequencedict[name[:-1]].append(wfmdict[fieldvalue])

    # clean the dict
    keys = list(sequencedict.keys())
    for key in keys:
        if sequencedict[key] == []:
            sequencedict.pop(key)

    # waveforms
    wfms = []
    m1s = []
    m2s = []
    channels = []
    for key in [key for key in sequencedict if 'WAVE' in key]:
        channels.append(_getendingnumber(key)[0])
        wfms_temp = []
        m1s_temp = []
        m2s_temp = []
        for wfmdict in sequencedict[key]:
            wfms_temp.append(wfmdict['wfm'])
            m1s_temp.append(wfmdict['m1'])
            m2s_temp.append(wfmdict['m2'])

        wfms.append(wfms_temp)
        m1s.append(m1s_temp)
        m2s.append(m2s_temp)

    nreps = sequencedict['SEQUENCE_LOOP']
    waits = sequencedict['SEQUENCE_WAIT']
    gotos = sequencedict['SEQUENCE_GOTO']
    jumps = sequencedict['SEQUENCE_JUMP']

    return (wfms, m1s, m2s, nreps, waits, gotos, jumps, channels)


def parse_awg_file(awgfilepath):
    """
    Parser for a binary .awg file. Returns a tuple matching the call signature
    of make_send_and_load_awg_file and a dictionary with instrument settings

    NOTE: Build-in waveforms are not stored in .awg files. Blame tektronix.

    Args:
        awgfilepath (str): The absolute path to the awg file

    Returns:
        tuple: (tuple, dict), where the first tuple is \
          (wfms, m1s, m2s, nreps, trigs, gotos, jumps, channels) \
          and the dict contains all instrument settings from the file
    """

    instdict, waveformlist, sequencelist = _parser1(awgfilepath)
    wfmdict = _parser2(waveformlist)
    callsigtuple = _parser3(sequencelist, wfmdict)

    return (callsigtuple, instdict)


