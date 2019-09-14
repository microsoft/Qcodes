"""
This module defines constants that are used in Alazar ATS API.

Since the original names of the constants are mostly preserved, it is
convenient to find the uesful constants here based on the Alazar SDK manual.
"""


from typing import NewType, Dict, Tuple
from enum import IntEnum, IntFlag


ReturnCode = NewType('ReturnCode', int)

API_SUCCESS = ReturnCode(512)
API_DMA_IN_PROGRESS = ReturnCode(518)

max_buffer_size = 64 * 1024 * 1024
# The maximum size of a single buffer
# in bytes. see docs of AlazarBeforeAsyncRead
# http://www.alazartech.com/Support/Download%20Files/ATS-SDK-Guide-7.2.3.pdf#section*.110

ERROR_CODES: Dict[ReturnCode, str] = {ReturnCode(code): msg for code, msg in {
    513: 'ApiFailed',
    514: 'ApiAccessDenied',
    515: 'ApiDmaChannelUnavailable',
    516: 'ApiDmaChannelInvalid',
    517: 'ApiDmaChannelTypeError',
    518: 'ApiDmaInProgress',
    519: 'ApiDmaDone',
    520: 'ApiDmaPaused',
    521: 'ApiDmaNotPaused',
    522: 'ApiDmaCommandInvalid',
    523: 'ApiDmaManReady',
    524: 'ApiDmaManNotReady',
    525: 'ApiDmaInvalidChannelPriority',
    526: 'ApiDmaManCorrupted',
    527: 'ApiDmaInvalidElementIndex',
    528: 'ApiDmaNoMoreElements',
    529: 'ApiDmaSglInvalid',
    530: 'ApiDmaSglQueueFull',
    531: 'ApiNullParam',
    532: 'ApiInvalidBusIndex',
    533: 'ApiUnsupportedFunction',
    534: 'ApiInvalidPciSpace',
    535: 'ApiInvalidIopSpace',
    536: 'ApiInvalidSize',
    537: 'ApiInvalidAddress',
    538: 'ApiInvalidAccessType',
    539: 'ApiInvalidIndex',
    540: 'ApiMuNotReady',
    541: 'ApiMuFifoEmpty',
    542: 'ApiMuFifoFull',
    543: 'ApiInvalidRegister',
    544: 'ApiDoorbellClearFailed',
    545: 'ApiInvalidUserPin',
    546: 'ApiInvalidUserState',
    547: 'ApiEepromNotPresent',
    548: 'ApiEepromTypeNotSupported',
    549: 'ApiEepromBlank',
    550: 'ApiConfigAccessFailed',
    551: 'ApiInvalidDeviceInfo',
    552: 'ApiNoActiveDriver',
    553: 'ApiInsufficientResources',
    554: 'ApiObjectAlreadyAllocated',
    555: 'ApiAlreadyInitialized',
    556: 'ApiNotInitialized',
    557: 'ApiBadConfigRegEndianMode',
    558: 'ApiInvalidPowerState',
    559: 'ApiPowerDown',
    560: 'ApiFlybyNotSupported',
    561: 'ApiNotSupportThisChannel',
    562: 'ApiNoAction',
    563: 'ApiHSNotSupported',
    564: 'ApiVPDNotSupported',
    565: 'ApiVpdNotEnabled',
    566: 'ApiNoMoreCap',
    567: 'ApiInvalidOffset',
    568: 'ApiBadPinDirection',
    569: 'ApiPciTimeout',
    570: 'ApiDmaChannelClosed',
    571: 'ApiDmaChannelError',
    572: 'ApiInvalidHandle',
    573: 'ApiBufferNotReady',
    574: 'ApiInvalidData',
    575: 'ApiDoNothing',
    576: 'ApiDmaSglBuildFailed',
    577: 'ApiPMNotSupported',
    578: 'ApiInvalidDriverVersion',
    579: ('ApiWaitTimeout: operation did not finish during '
          'timeout interval. Check your trigger.'),
    580: 'ApiWaitCanceled',
    581: 'ApiBufferTooSmall',
    582: ('ApiBufferOverflow: rate of acquiring data > rate of '
          'transferring data to local memory. Try reducing sample rate, '
          'reducing number of enabled channels, increasing size of each '
          'DMA buffer or increase number of DMA buffers.'),
    583: 'ApiInvalidBuffer',
    584: 'ApiInvalidRecordsPerBuffer',
    585: ('ApiDmaPending: Async I/O operation was successfully started, '
          'it will be completed when sufficient trigger events are '
          'supplied to fill the buffer.'),
    586: ('ApiLockAndProbePagesFailed:Driver or operating system was '
          'unable to prepare the specified buffer for DMA transfer. '
          'Try reducing buffer size or total number of buffers.'),
    587: 'ApiWaitAbandoned',
    588: 'ApiWaitFailed',
    589: ('ApiTransferComplete: This buffer is last in the current '
          'acquisition.'),
    590: 'ApiPllNotLocked: hardware error, contact AlazarTech',
    591: ('ApiNotSupportedInDualChannelMode:Requested number of samples '
          'per channel is too large to fit in on-board memory. Try '
          'reducing number of samples per channel, or switch to '
          'single channel mode.'),
    592: ('ApiNotSupportedInQuadChannelMode: The requested number of '
          'samples per channel is too large to fit in onboard memory. '
          'Try reducing the num'),
    593: 'ApiFileIoError: A file read or write error occurred.',
    594: ('ApiInvalidClockFrequency: The requested ADC clock frequency is '
          'not supported.'),
    595: 'ApiInvalidSkipTable',
    596: 'ApiInvalidDspModule',
    597: 'ApiDESOnlySupportedInSingleChannelMode',
    598: 'ApiInconsistentChannel',
    599: 'ApiDspFiniteRecordsPerAcquisition',
    600: 'ApiNotEnoughNptFooters',
    601: 'ApiInvalidNptFooter',
    602: ('ApiOCTIgnoreBadClockNotSupported: OCT ignore bad clock is not '
          'supported.'),
    603: ('ApiError: The requested number of records in a single-port '
          'acquisition exceeds the maximum supported by the digitizer. '
          'Use dual-ported AutoDMA to acquire more records per acquisition.'),
    604: ('ApiError: The requested number of records in a single-port '
          'acquisition exceeds the maximum supported by the digitizer.'),
    605: ('ApiOCTNoTriggerDetected: No trigger is detected for OCT ignore '
          'bad clock feature.'),
    606: ('ApiOCTTriggerTooFast: Trigger detected is too fast for OCT ignore '
          'bad clock feature.'),
    607: ('ApiNetworkError: There was an issue related to network. Make sure '
          'that the network connection and settings are correct.'),
    608: ('ApiFftSizeTooLarge: The on-FPGA FFT cannot support FFT that '
          'large. Try reducing the FFT size.'),
}.items()}


BOARD_NAMES = {
    0: 'ATS_NONE',
    1: 'ATS850',
    2: 'ATS310',
    3: 'ATS330',
    4: 'ATS855',
    5: 'ATS315',
    6: 'ATS335',
    7: 'ATS460',
    8: 'ATS860',
    9: 'ATS660',
    10: 'ATS665',
    11: 'ATS9462',
    12: 'ATS9434',
    13: 'ATS9870',
    14: 'ATS9350',
    15: 'ATS9325',
    16: 'ATS9440',
    17: 'ATS9410',
    18: 'ATS9351',
    19: 'ATS9310',
    20: 'ATS9461',
    21: 'ATS9850',
    22: 'ATS9625',
    23: 'ATG6500',
    24: 'ATS9626',
    25: 'ATS9360',
    26: 'AXI9870',
    27: 'ATS9370',
    28: 'ATU7825',
    29: 'ATS9373',
    30: 'ATS9416',
    31: 'ATS9637',
    32: 'ATS9120',
    33: 'ATS9371',
    34: 'ATS9130',
    35: 'ATS9352',
    36: 'ATS9453',
}


# See table 4 Input Channel Configurations
# on page 241 of the Alazar SDK manual
_NUMBER_OF_CHANNELS_TO_BYTE_REPR: Dict[int, Tuple[int, ...]] = {
    1:  tuple(2**i for i in range(16)),
    2:  (3, 5, 6, 9, 10, 12),
    4:  (15,),
    8:  (255,),
    16: (65535,)
    }

# See table 4 Input Channel Configurations
# on page 241 of the Alazar SDK manual
NUMBER_OF_CHANNELS_FROM_BYTE_REPR: Dict[int, int] = {
    byte_repr: n_ch
        for n_ch, byte_reprs in _NUMBER_OF_CHANNELS_TO_BYTE_REPR.items()
        for byte_repr in byte_reprs
    }


REGISTER_ACCESS_PASSWORD = 0x32145876


class ClockSource(IntEnum):
    INTERNAL_CLOCK = 0x00000001
    FAST_EXTERNAL_CLOCK = 0x00000002
    EXTERNAL_CLOCK = FAST_EXTERNAL_CLOCK
    MEDIUM_EXTERNAL_CLOCK = 0x00000003
    SLOW_EXTERNAL_CLOCK = 0x00000004
    EXTERNAL_CLOCK_AC = 0x00000005
    EXTERNAL_CLOCK_DC = 0x00000006
    EXTERNAL_CLOCK_10MHz_REF = 0x00000007
    INTERNAL_CLOCK_10MHz_REF = 0x00000008
    EXTERNAL_CLOCK_10MHz_PXI = 0x0000000A
    INTERNAL_CLOCK_DIV_4 = 0x0000000F
    INTERNAL_CLOCK_DIV_5 = 0x00000010
    MASTER_CLOCK = 0x00000011
    INTERNAL_CLOCK_SET_VCO = 0x00000012


class ClockEdge(IntEnum):
    CLOCK_EDGE_RISING = 0x00000000
    CLOCK_EDGE_FALLING = 0x00000001


class SampleRate(IntEnum):
    SAMPLE_RATE_USER_DEF = 0x00000040
    SAMPLE_RATE_1KSPS = 0X00000001
    SAMPLE_RATE_2KSPS = 0X00000002
    SAMPLE_RATE_5KSPS = 0X00000004
    SAMPLE_RATE_10KSPS = 0X00000008
    SAMPLE_RATE_20KSPS = 0X0000000A
    SAMPLE_RATE_50KSPS = 0X0000000C
    SAMPLE_RATE_100KSPS = 0X0000000E
    SAMPLE_RATE_200KSPS = 0X00000010
    SAMPLE_RATE_500KSPS = 0X00000012
    SAMPLE_RATE_1MSPS = 0X00000014
    SAMPLE_RATE_2MSPS = 0X00000018
    SAMPLE_RATE_5MSPS = 0X0000001A
    SAMPLE_RATE_10MSPS = 0X0000001C
    SAMPLE_RATE_20MSPS = 0X0000001E
    SAMPLE_RATE_25MSPS = 0X00000021
    SAMPLE_RATE_50MSPS = 0X00000022
    SAMPLE_RATE_100MSPS = 0X00000024
    SAMPLE_RATE_125MSPS = 0x00000025
    SAMPLE_RATE_160MSPS = 0x00000026
    SAMPLE_RATE_180MSPS = 0x00000027
    SAMPLE_RATE_200MSPS = 0X00000028
    SAMPLE_RATE_250MSPS = 0X0000002B
    SAMPLE_RATE_400MSPS = 0X0000002D
    SAMPLE_RATE_500MSPS = 0X00000030
    SAMPLE_RATE_800MSPS = 0X00000032
    SAMPLE_RATE_1000MSPS = 0x00000035
    SAMPLE_RATE_1GSPS = SAMPLE_RATE_1000MSPS
    SAMPLE_RATE_1200MSPS = 0x00000037
    SAMPLE_RATE_1500MSPS = 0x0000003A
    SAMPLE_RATE_1600MSPS = 0x0000003B
    SAMPLE_RATE_1800MSPS = 0x0000003D
    SAMPLE_RATE_2000MSPS = 0x0000003F
    SAMPLE_RATE_2GSPS = SAMPLE_RATE_2000MSPS
    SAMPLE_RATE_2400MSPS = 0x0000006A
    SAMPLE_RATE_3000MSPS = 0x00000075
    SAMPLE_RATE_3GSPS = SAMPLE_RATE_3000MSPS
    SAMPLE_RATE_3600MSPS = 0x0000007B
    SAMPLE_RATE_4000MSPS = 0x00000080
    SAMPLE_RATE_4GSPS = SAMPLE_RATE_4000MSPS


class Coupling(IntEnum):
    AC_COUPLING = 0x00000001
    DC_COUPLING = 0x00000002


class InputRange(IntEnum):
    INPUT_RANGE_PM_20_MV = 0x00000001
    INPUT_RANGE_PM_40_MV = 0x00000002
    INPUT_RANGE_PM_50_MV = 0x00000003
    INPUT_RANGE_PM_80_MV = 0x00000004
    INPUT_RANGE_PM_100_MV = 0x00000005
    INPUT_RANGE_PM_200_MV = 0x00000006
    INPUT_RANGE_PM_400_MV = 0x00000007
    INPUT_RANGE_PM_500_MV = 0x00000008
    INPUT_RANGE_PM_800_MV = 0x00000009
    INPUT_RANGE_PM_1_V = 0x0000000A
    INPUT_RANGE_PM_2_V = 0x0000000B
    INPUT_RANGE_PM_4_V = 0x0000000C
    INPUT_RANGE_PM_5_V = 0x0000000D
    INPUT_RANGE_PM_8_V = 0x0000000E
    INPUT_RANGE_PM_10_V = 0x0000000F
    INPUT_RANGE_PM_20_V = 0x00000010
    INPUT_RANGE_PM_40_V = 0x00000011
    INPUT_RANGE_PM_16_V = 0x00000012
    INPUT_RANGE_HIFI = 0x00000020
    INPUT_RANGE_PM_1_V_25 = 0x00000021
    INPUT_RANGE_PM_2_V_5 = 0x00000025
    INPUT_RANGE_PM_125_MV = 0x00000028
    INPUT_RANGE_PM_250_MV = 0x00000030


class Impedance(IntEnum):
    IMPEDANCE_1M_OHM = 0x00000001
    IMPEDANCE_50_OHM = 0x00000002
    IMPEDANCE_75_OHM = 0x00000004
    IMPEDANCE_300_OHM = 0x00000008


class BandwidthLimit(IntEnum):
    DISABLE = 0
    ENABLE = 1


class TriggerEngine(IntEnum):
    TRIG_ENGINE_J = 0x00000000
    TRIG_ENGINE_K = 0x00000001


class TriggerEngineOperation(IntEnum):
    TRIG_ENGINE_OP_J = 0x00000000
    TRIG_ENGINE_OP_K = 0x00000001
    TRIG_ENGINE_OP_J_OR_K = 0x00000002
    TRIG_ENGINE_OP_J_AND_K = 0x00000003
    TRIG_ENGINE_OP_J_XOR_K = 0x00000004
    TRIG_ENGINE_OP_J_AND_NOT_K = 0x00000005
    TRIG_ENGINE_OP_NOT_J_AND_K = 0x00000006


class TriggerEngineSource(IntEnum):
    TRIG_CHAN_A = 0x00000000
    TRIG_CHAN_B = 0x00000001
    TRIG_EXTERNAL = 0x00000002
    TRIG_DISABLE = 0x00000003
    TRIG_CHAN_C = 0x00000004
    TRIG_CHAN_D = 0x00000005
    TRIG_CHAN_E = 0x00000006
    TRIG_CHAN_F = 0x00000007
    TRIG_CHAN_G = 0x00000008
    TRIG_CHAN_H = 0x00000009
    TRIG_CHAN_I = 0x0000000A
    TRIG_CHAN_J = 0x0000000B
    TRIG_CHAN_K = 0x0000000C
    TRIG_CHAN_L = 0x0000000D
    TRIG_CHAN_M = 0x0000000E
    TRIG_CHAN_N = 0x0000000F
    TRIG_CHAN_O = 0x00000010
    TRIG_CHAN_P = 0x00000011
    TRIG_PXI_STAR = 0x00000100


class TriggerSlope(IntEnum):
    """
    Used in more than one place, for example, for ``AUX_IN_TRIGGER_ENABLE``
    Auxiallary IO mode
    """
    TRIGGER_SLOPE_POSITIVE = 0x00000001
    TRIGGER_SLOPE_NEGATIVE = 0x00000002


class ExternalTriggerCoupling(IntEnum):
    AC = 1
    DC = 2


class ExternalTriggerRange(IntEnum):
    ETR_DIV5 = 0x00000000
    ETR_X1 = 0x00000001
    ETR_5V = 0x00000000
    ETR_1V = 0x00000001
    ETR_TTL = 0x00000002
    ETR_2V5 = 0x00000003


# This is how it's called in AlazarCmd.h
ExternalTriggerAttenuatorRelay = ExternalTriggerRange


class AuxilaryIO(IntEnum):
    # Output
    AUX_OUT_TRIGGER = 0
    AUX_OUT_PACER = 2
    AUX_OUT_BUSY = 4
    AUX_OUT_CLOCK = 6
    AUX_OUT_RESERVED = 8
    AUX_OUT_CAPTURE_ALMOST_DONE = 10
    AUX_OUT_AUXILIARY = 12
    AUX_OUT_SERIAL_DATA = 14
    AUX_OUT_TRIGGER_ENABLE = 16
    # Input
    AUX_IN_TRIGGER_ENABLE = 1
    AUX_IN_DIGITAL_TRIGGER = 3
    AUX_IN_GATE = 5
    AUX_IN_CAPTURE_ON_DEMAND = 7
    AUX_IN_RESET_TIMESTAMP = 9
    AUX_IN_SLOW_EXTERNAL_CLOCK = 11
    AUX_IN_AUXILIARY = 13
    AUX_IN_SERIAL_DATA = 15


class AutoDMAFlag(IntFlag):
    # Automatic DMA acquisition modes
    ADMA_TRADITIONAL_MODE = 0x00000000
    ADMA_CONTINUOUS_MODE = 0x00000100
    ADMA_NPT = 0x00000200
    ADMA_TRIGGERED_STREAMING = 0x00000400
    # Automatic DMA acquisition options
    ADMA_EXTERNAL_STARTCAPTURE = 0x00000001
    ADMA_ENABLE_RECORD_HEADERS = 0x00000008
    ADMA_FIFO_ONLY_STREAMING = 0x00000800
    ADMA_ALLOC_BUFFERS = 0x00000020
    ADMA_INTERLEAVE_SAMPLES = 0x00001000
    ADMA_GET_PROCESSED_DATA = 0x00002000
    ADMA_DSP = 0x00004000
    # Other
    ADMA_SINGLE_DMA_CHANNEL = 0x00000010
    ADMA_ENABLE_RECORD_FOOTERS = 0x00010000


class Channel(IntFlag):
    """Flags for selecting channel configuration"""
    ALL = 0x00000000
    A = 0x00000001
    B = 0x00000002
    C = 0x00000004
    D = 0x00000008
    E = 0x00000010
    F = 0x00000020
    G = 0x00000040
    H = 0x00000080
    I = 0x00000100
    J = 0x00000200
    K = 0x00000400
    L = 0x00000800
    M = 0x00001000
    N = 0x00002000
    O = 0x00004000
    P = 0x00008000


class Capability(IntEnum):
    """Capability identifiers for :meth:`.AlazarATSAPI.query_capability`"""
    GET_SERIAL_NUMBER = 0x10000024
    # Date of the board's latest calibration data as a decimal number with
    # the format DDMMYY where DD is 1-31, MM is 1-12,
    # and YY is 00-99 from 2000
    GET_LATEST_CAL_DATE = 0x10000026
    # Month of the board's latest calibration date as
    # a decimal number with the format MM where M is 1-12
    GET_LATEST_CAL_DATE_MONTH = 0x1000002D
    # Day of month of the board's latest calibration date
    # as a decimal number with the format DD where DD is 1-31
    GET_LATEST_CAL_DATE_DAY = 0x1000002E
    # Year of the board's latest calibration date
    # as a decimal number with the format YY where YY is 00-99 from 2000
    GET_LATEST_CAL_DATE_YEAR = 0x1000002F
    # On-board memory size in maximum samples per channel in single channel
    # mode; see AlazarGetChannelInfo for more information
    MEMORY_SIZE = 0x1000002A
    # Board's FPGA signature
    ASOPC_TYPE = 0x1000002C
    # The PCIe link speed negotiated between a PCIe digitizer board and
    # the host PCIe bus in 2.5G bits per second units. The PCIe bus uses
    # 10b/8b encoding, so divide the link speed by 10 to find the link speed
    # in bytes per second. For example, a link speed of 2.5 Gb/s gives
    # 250 MB/s per lane. PCIe Gen 2 digitizers such as ATS9360 should receive
    # 5.0 Gb/s links. PCIe Gen 3 digitizers such as ATS9373 should receive
    # 8.0 Gb/s links.
    GET_PCIE_LINK_SPEED = 0x10000030
    # The PCIe link width in lanes negociated between a PCIe digitizer board
    # and the host PCIe bus. An ATS9462 should negociate 4 lanes, while the
    # ATS9325, ATS9350, ATS9351, ATS9360, ATS9373, ATS9440, ATS9850 and
    # ATS9870 should negociate 8 lanes. If a board obtains fewer lanes, then
    # the board may be installed in a PCIe slot that does not support
    # the expected number of lanes. The ideal PCIe bandwidth is the link speed
    # in bytes per second per lane, multiplied by the link width in lanes.
    # For example, and ATS9870 that negociates 8 lanes at 250 MB/s per lane
    # has an ideal bandwidth of 2 GB/s.
    GET_PCIE_LINK_WIDTH = 0x10000031
    # Board type identifier; see AlazarGetBoardKind for more information
    BOARD_TYPE = 0x1000002B
    # Get the maximum number of pre-trigger samples
    GET_MAX_PRETRIGGER_SAMPLES = 0x10000046
    # The model of user-programable FPGA device (see also `CPFDevice`)
    GET_CPF_DEVICE = 0x10000071


CPF_DEVICE_EP3SL50 = 1
CPF_DEVICE_EP3SE260 = 2


class CPFDevice(IntEnum):
    EP3SL50 = CPF_DEVICE_EP3SL50
    EP3SE260 = CPF_DEVICE_EP3SE260


CRA_MODE_DISABLE = 0
CRA_MODE_ENABLE_FPGA_AVE = 1


class RecordAverageMode(IntEnum):
    """
    Values for ``mode`` argument of ``AlazarConfigureRecordAverage`` function
    """
    DISABLE = CRA_MODE_DISABLE
    ENABLE_FPGA_AVE = CRA_MODE_ENABLE_FPGA_AVE


CRA_OPTION_UNSIGNED = 0
CRA_OPTION_SIGNED = 1


class RecordAverageOption(IntEnum):
    """
    Values for ``options`` argument of ``AlazarConfigureRecordAverage``
    function
    """
    UNSIGNED = CRA_OPTION_UNSIGNED  #: Find sum of unsigned ADC samples codes
    SIGNED = CRA_OPTION_SIGNED  #: Find sum of signed ADC sample


LED_OFF = 0
LED_ON = 1


class AlazarParameter(IntEnum):
    """
    Parameters suitable to be used with `` AlazarSetParameter`` and/or
    ``AlazarGetParameter``
    Defined by ``ALAZAR_PARAMETERS`` in ``AlazarCmd.h``
    """
    DATA_WIDTH = 0x10000009
    # The number of bits per sample
    SETGET_ASYNC_BUFFSIZE_BYTES = 0x10000039
    # The size of API-allocated DMA buffers in bytes
    SETGET_ASYNC_BUFFCOUNT = 0x10000040
    # The number of API-allocated DMA buffers
    GET_ASYNC_BUFFERS_PENDING = 0x10000050
    # DMA buffers currently posted to the board
    GET_ASYNC_BUFFERS_PENDING_FULL = 0x10000051
    # DMA buffers waiting to be processed by the application
    GET_ASYNC_BUFFERS_PENDING_EMPTY = 0x10000052
    # DMA buffers waiting to be filled by the board
    SET_DATA_FORMAT = 0x10000041
    # 0 if the data format is unsigned, and 1 otherwise
    GET_DATA_FORMAT = 0x10000042
    # 0 if the data format is unsigned, and 1 otherwise
    GET_SAMPLES_PER_TIMESTAMP_CLOCK = 0x10000044
    # Number of samples per timestamp clock
    GET_RECORDS_CAPTURED = 0x10000045
    # Records captured since the start of the acquisition (single-port)
    # or buffer (dual-port)
    ECC_MODE = 0x10000048
    # ECC mode. Member of ECCMode enum
    GET_AUX_INPUT_LEVEL = 0x10000049
    # Read the TTL level of the AUX connector.
    # Member of  AUXInputLevel enum
    GET_CHANNELS_PER_BOARD = 0x10000070
    # Number of analog channels supported by this digitizer
    GET_FPGA_TEMPERATURE = 0x10000080
    # Current FPGA temperature in degrees Celcius. Only supported by
    # PCIe digitizers.
    PACK_MODE = 0x10000072
    # Get/Set the pack mode as a member of PackMode enum
    SET_SINGLE_CHANNEL_MODE = 0x10000043
    # Reserve all the on-board memory to the channel passed as
    # argument. Single-port only.
    API_FLAGS = 0x10000090
    # State of the API logging as a member of
    # API_TRACE_STATES enum


class ECCMode(IntEnum):
    """
    Values for ECC_MODE of ``Parameter``
    Defined by ``ALAZAR_ECC_MODES`` in ``AlazarCmd.h``
    """
    ECC_DISABLE = 0  # Disable
    ECC_ENABLE = 1  # Enable


class PackMode(IntEnum):
    """
    Values for PACK_MODE of ``Parameter``
    Defined by ``ALAZAR_PACK_MODES`` in ``AlazarCmd.h``
    """
    PACK_DEFAULT = 0  # Default pack mode of the board
    PACK_8_BITS_PER_SAMPLE = 1  # 8 bits per sample
    PACK_12_BITS_PER_SAMPLE = 2  # 12 bits per sample


class AUXInputLevel(IntEnum):
    """
    Values for GET_AUX_INPUT_LEVEL of ``Parameter``
    Defined by ``ALAZAR_AUX_INPUT_LEVELS`` in ``AlazarCmd.h``
    """
    AUX_INPUT_LOW = 0  # Low level
    AUX_INPUT_HIGH = 1  # High level


class APITraceStates(IntEnum):
    """
    Values for API_FLAGS of ``Parameter``
    Defined by ``ALAZAR_API_TRACE_STATES`` in ``AlazarCmd.h``
    """
    API_ENABLE_TRACE = 1  # Trace enabled
    API_DISABLE_TRACE = 0  # Trace disabled
