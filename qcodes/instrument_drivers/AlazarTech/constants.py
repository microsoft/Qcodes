from typing import NewType, Dict
from enum import IntEnum


ReturnCode = NewType('ReturnCode', int)

API_SUCCESS = 512

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


REGISTER_READING_PWD = 0x32145876


# Capability identifiers

class Capability(IntEnum):
    """Capability identifiers for 'query capability' function"""
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
