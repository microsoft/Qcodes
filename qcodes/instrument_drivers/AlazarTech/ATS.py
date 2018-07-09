import ctypes
import logging
import numpy as np
import time
import os
import warnings
import sys

from typing import List, Dict, Union, Tuple, cast, Sequence
from contextlib import contextmanager

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
from .utils import TraceParameter

logger = logging.getLogger(__name__)

# TODO(damazter) (C) logging

# these items are important for generalizing this code to multiple alazar cards
# TODO(damazter) (W) some alazar cards have a different number of channels :(

# TODO(damazter) (S) tests to do:
# acquisition that would overflow the board if measurement is not stopped
# quickly enough. can this be solved by not reposting the buffers?

# TODO (natalie) make logging vs print vs nothing decisions


class AlazarTech_ATS(Instrument):
    """
    This is the qcodes driver for Alazar data acquisition cards

    status: beta-version

    this driver is written with the ATS9870 in mind
    updates might/will be necessary for other versions of Alazar cards

    Args:

        name: name for this instrument, passed to the base instrument
        system_id: target system id for this instrument
        board_id: target board id within the system for this instrument
        dll_path: string containing the path of the ATS driver dll

    """
    # override dll_path in your init script or in the board constructor
    # if you have it somewhere else
    dll_path = 'C:\\WINDOWS\\System32\\ATSApi'

    # override channels in a subclass if needed
    channels = 2

    _success = 512

    _error_codes = {
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
        582: ('ApiBufferOverflow:rate of acquiring data > rate of '
              'transferring data to local memory. Try reducing sample rate, '
              'reducing number of enabled channels, increasing size of each '
              'DMA buffer or increase number of DMA buffers.'),
        583: 'ApiInvalidBuffer',
        584: 'ApiInvalidRecordsPerBuffer',
        585: ('ApiDmaPending:Async I/O operation was successfully started, '
              'it will be completed when sufficient trigger events are '
              'supplied to fill the buffer.'),
        586: ('ApiLockAndProbePagesFailed:Driver or operating system was '
              'unable to prepare the specified buffer for DMA transfer. '
              'Try reducing buffer size or total number of buffers.'),
        587: 'ApiWaitAbandoned',
        588: 'ApiWaitFailed',
        589: ('ApiTransferComplete:This buffer is last in the current '
              'acquisition.'),
        590: 'ApiPllNotLocked:hardware error, contact AlazarTech',
        591: ('ApiNotSupportedInDualChannelMode:Requested number of samples '
              'per channel is too large to fit in on-board memory. Try '
              'reducing number of samples per channel, or switch to '
              'single channel mode.')
    }

    _board_names = {
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
        30: 'ATS9416'
    }

    @classmethod
    def find_boards(cls, dll_path: str=None) -> List[dict]:
        """
        Find Alazar boards connected

        Args:
            dll_path: (string) path of the Alazar driver dll

        Returns:
            list: list of board info for each connected board
        """
        dll = ctypes.cdll.LoadLibrary(dll_path or cls.dll_path)

        system_count = dll.AlazarNumOfSystems()
        boards = []
        for system_id in range(1, system_count + 1):
            board_count = dll.AlazarBoardsInSystemBySystemID(system_id)
            for board_id in range(1, board_count + 1):
                boards.append(cls.get_board_info(dll, system_id, board_id))
        return boards

        # TODO(nataliejpg) this needs fixing..., dll can't be a string
    @classmethod
    def get_board_info(cls, dll: ctypes.CDLL, system_id: int,
                       board_id: int) -> Dict[str,Union[str,int]]:
        """
        Get the information from a connected Alazar board

        Args:
            dll (CDLL): CTypes CDLL
            system_id: id of the Alazar system
            board_id: id of the board within the alazar system

        Return:

            Dictionary containing

                - system_id
                - board_id
                - board_kind (as string)
                - max_samples
                - bits_per_sample
        """
        # make a temporary instrument for this board, to make it easier
        # to get its info
        board = cls('temp', system_id=system_id, board_id=board_id,
                    server_name=None)

        handle = board._handle
        board_kind = cls._board_names[dll.AlazarGetBoardKind(handle)]

        max_s, bps = board._get_channel_info(handle)
        return {
            'system_id': system_id,
            'board_id': board_id,
            'board_kind': board_kind,
            'max_samples': max_s,
            'bits_per_sample': bps
        }

    def __init__(self, name: str, system_id: int=1, board_id: int=1,
                 dll_path: str=None, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self._ATS_dll = None

        if os.name == 'nt':
            self._ATS_dll = ctypes.cdll.LoadLibrary(dll_path or self.dll_path)
        else:
            raise Exception("Unsupported OS")
        self._parameters_synced = False
        self._handle = self._ATS_dll.AlazarGetBoardBySystemID(system_id,
                                                              board_id)
        if not self._handle:
            raise Exception('AlazarTech_ATS not found at '
                            'system {}, board {}'.format(system_id, board_id))

        self.buffer_list: List['Buffer'] = []

        self._ATS_dll.AlazarWaitAsyncBufferComplete.argtypes = [
            ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
        self._ATS_dll.AlazarBeforeAsyncRead.argtypes = [ctypes.c_uint32,
                                                        ctypes.c_uint32,
                                                        ctypes.c_long,
                                                        ctypes.c_uint32,
                                                        ctypes.c_uint32,
                                                        ctypes.c_uint32,
                                                        ctypes.c_uint32]
        self._ATS_dll.AlazarSetCaptureClock.argtypes = [ctypes.c_uint32,
                                                        ctypes.c_uint32,
                                                        ctypes.c_uint32,
                                                        ctypes.c_uint32,
                                                        ctypes.c_uint32]
        self._ATS_dll.AlazarPostAsyncBuffer.argtypes = [ctypes.c_uint32,
                                                        ctypes.c_void_p,
                                                        ctypes.c_uint32]
        if sys.platform == 'win32':
            ctypes.windll.kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p,
                                                            ctypes.c_long,
                                                            ctypes.c_long,
                                                            ctypes.c_long]
            ctypes.windll.kernel32.VirtualFree.argtypes = [ctypes.c_void_p,
                                                           ctypes.c_long,
                                                           ctypes.c_long]

    def get_idn(self) -> dict:
        # TODO this is really Dict[str, Optional[Union[str,int]]]
        # But that is inconsistent with the super class. We should consider
        # if ints and floats are allowed as values in the dict
        """
        This methods gets the most relevant information of this instrument

        Returns:

            Dictionary containing

                - 'firmware': None
                - 'model': as string
                - 'serial': board serial number
                - 'vendor': 'AlazarTech',
                - 'CPLD_version': version of the CPLD
                - 'driver_version': version of the driver dll
                - 'SDK_version': version of the SDK
                - 'latest_cal_date': date of the latest calibration (as string)
                - 'memory_size': size of the memory in samples,
                - 'asopc_type': type of asopc (as decimal number),
                - 'pcie_link_speed': the speed of a single pcie link (in GB/s),
                - 'pcie_link_width': number of pcie links
        """
        board_kind = self._board_names[
            self._ATS_dll.AlazarGetBoardKind(self._handle)]

        max_s, bps = self._get_channel_info(self._handle)

        major = ctypes.c_uint8(0)
        minor = ctypes.c_uint8(0)
        revision = ctypes.c_uint8(0)
        self._call_dll('AlazarGetCPLDVersion',
                       self._handle,
                       ctypes.byref(major),
                       ctypes.byref(minor))
        cpld_ver = str(major.value) + "." + str(minor.value)

        self._call_dll('AlazarGetDriverVersion',
                       ctypes.byref(major),
                       ctypes.byref(minor),
                       ctypes.byref(revision))
        driver_ver = str(major.value)+"."+str(minor.value)+"."+str(revision.value)

        self._call_dll('AlazarGetSDKVersion',
                       ctypes.byref(major),
                       ctypes.byref(minor),
                       ctypes.byref(revision))
        sdk_ver = str(major.value)+"."+str(minor.value)+"."+str(revision.value)

        value = ctypes.c_uint32(0)
        self._call_dll('AlazarQueryCapability',
                       self._handle, 0x10000024, 0, ctypes.byref(value))
        serial = str(value.value)
        self._call_dll('AlazarQueryCapability',
                       self._handle, 0x10000026, 0, ctypes.byref(value))
        Capabilitystring = str(value.value)
        latest_cal_date = (Capabilitystring[0:2] + "-" +
                           Capabilitystring[2:4] + "-" +
                           Capabilitystring[4:6])

        self._call_dll('AlazarQueryCapability',
                       self._handle, 0x1000002A, 0, ctypes.byref(value))
        memory_size = str(value.value)
        self._call_dll('AlazarQueryCapability',
                       self._handle, 0x1000002C, 0, ctypes.byref(value))
        asopc_type = str(value.value)

        # see the ATS-SDK programmer's guide
        # about the encoding of the link speed
        self._call_dll('AlazarQueryCapability',
                       self._handle, 0x10000030, 0,  ctypes.byref(value))
        pcie_link_speed = str(value.value * 2.5 / 10) + "GB/s"
        self._call_dll('AlazarQueryCapability',
                       self._handle, 0x10000031, 0, ctypes.byref(value))
        pcie_link_width = str(value.value)

        return {'firmware': None,
                'model': board_kind,
                'max_samples': max_s,
                'bits_per_sample': bps,
                'serial': serial,
                'vendor': 'AlazarTech',
                'CPLD_version': cpld_ver,
                'driver_version': driver_ver,
                'SDK_version': sdk_ver,
                'latest_cal_date': latest_cal_date,
                'memory_size': memory_size,
                'asopc_type': asopc_type,
                'pcie_link_speed': pcie_link_speed,
                'pcie_link_width': pcie_link_width}

    def config(self,
               clock_source=None, sample_rate=None, clock_edge=None,
               external_sample_rate=None,
               decimation=None, coupling=None, channel_range=None,
               impedance=None, bwlimit=None, trigger_operation=None,
               trigger_engine1=None, trigger_source1=None,
               trigger_slope1=None, trigger_level1=None,
               trigger_engine2=None, trigger_source2=None,
               trigger_slope2=None, trigger_level2=None,
               external_trigger_coupling=None, external_trigger_range=None,
               trigger_delay=None, timeout_ticks=None, aux_io_mode=None,
               aux_io_param=None) -> None:
        """
        configure the ATS board and set the corresponding parameters to the
        appropriate values.
        For documentation of the parameters, see ATS-SDK programmer's guide

        Args:
            clock_source:
            sample_rate:
            clock_edge:
            external_sample_rate:
            decimation:
            coupling:
            channel_range:
            impedance:
            bwlimit:
            trigger_operation:
            trigger_engine1:
            trigger_source1:
            trigger_slope1:
            trigger_level1:
            trigger_engine2:
            trigger_source2:
            trigger_slope2:
            trigger_level2:
            external_trigger_coupling:
            external_trigger_range:
            trigger_delay:
            timeout_ticks:
            aux_io_mode:
            aux_io_param:

        Returns:
            None
        """
        # region set parameters from args
        warnings.warn("Alazar config is deprecated. Please replace with setting "
                      "of paramters directly with the syncing context manager",
                      stacklevel=2)
        self._set_if_present('clock_source', clock_source)
        self._set_if_present('sample_rate', sample_rate)
        self._set_if_present('external_sample_rate', external_sample_rate)
        self._set_if_present('clock_edge', clock_edge)
        self._set_if_present('decimation', decimation)

        self._set_list_if_present('coupling', coupling)
        self._set_list_if_present('channel_range', channel_range)
        self._set_list_if_present('impedance', impedance)
        self._set_list_if_present('bwlimit', bwlimit)

        self._set_if_present('trigger_operation', trigger_operation)
        self._set_if_present('trigger_engine1', trigger_engine1)
        self._set_if_present('trigger_source1', trigger_source1)
        self._set_if_present('trigger_slope1', trigger_slope1)
        self._set_if_present('trigger_level1', trigger_level1)

        self._set_if_present('trigger_engine2', trigger_engine2)
        self._set_if_present('trigger_source2', trigger_source2)
        self._set_if_present('trigger_slope2', trigger_slope2)
        self._set_if_present('trigger_level2', trigger_level2)

        self._set_if_present('external_trigger_coupling',
                             external_trigger_coupling)
        self._set_if_present('external_trigger_range',
                             external_trigger_range)
        self._set_if_present('trigger_delay', trigger_delay)
        self._set_if_present('timeout_ticks', timeout_ticks)
        self._set_if_present('aux_io_mode', aux_io_mode)
        self._set_if_present('aux_io_param', aux_io_param)
        # endregion

        # handle that external clock and internal clock uses
        # two different ways of setting the sample rate.
        # We use the matching one and mark the order one
        # as up to date since it's not being pushed to
        # the instrument at any time and is never used
        if sample_rate is not None and external_sample_rate is not None:
            raise RuntimeError("Both sample_rate and external_sample_rate supplied")

        if clock_source == 'EXTERNAL_CLOCK_10MHz_REF':
            if sample_rate is not None:
                logger.warning("Using external 10 MHz ref clock "
                               "but internal sample rate supplied. "
                               "Please use 'external_sample_rate'")
            sample_rate = self.external_sample_rate
        elif clock_source == 'INTERNAL_CLOCK':
            sample_rate = self.sample_rate
            if external_sample_rate is not None:
                logger.warning("Using internal clock "
                               "but external sample rate supplied. "
                               "Please use 'external_sample_rate'")

        self.sync_settings_to_card()

    @contextmanager
    def syncing(self):
        """
        Context manager for syncing settings to Alazar card. It will
        automatically call sync_settings_to_card at the end of the
        context.

        Example:
            This is intended to be used around multiple parameter sets
            to ensure syncing is done exactly once::

                with alazar.syncing():
                     alazar.trigger_source1('EXTERNAL')
                     alazar.trigger_level1(100)
        """

        yield
        self.sync_settings_to_card()

    def sync_settings_to_card(self) -> None:
        """
        Syncs all parameters to Alazar card
        """
        if self.clock_source() == 'EXTERNAL_CLOCK_10MHz_REF':
            sample_rate = self.external_sample_rate
            if self.external_sample_rate.raw_value == 'UNDEFINED':
                raise RuntimeError("Using external 10 MHz Ref but external "
                                   "sample_rate is not set")
            if self.sample_rate.raw_value != 'UNDEFINED':
                warnings.warn("Using external 10 MHz Ref but parameter sample_"
                              "rate is set. This will have no effect and "
                              "is ignored")
            # mark the unused parameter as up to date
            self.sample_rate._set_updated()
        else:
            if self.sample_rate.raw_value == 'UNDEFINED':
                raise RuntimeError("Using Internal clock but parameter sample_rate is not set")
            if self.external_sample_rate.raw_value != 'UNDEFINED':
                warnings.warn("Using Internal clock but parameter external_sample_rate is set."
                              "This will have no effect and is ignored")
            # mark the unused parameter as up to date
            self.external_sample_rate._set_updated()
            sample_rate = self.sample_rate

        self._call_dll('AlazarSetCaptureClock',
                       self._handle, self.clock_source, sample_rate,
                       self.clock_edge, self.decimation)

        for i in range(1, self.channels + 1):
            self._call_dll('AlazarInputControl',
                           self._handle, 2**(i-1),
                           self.parameters['coupling' + str(i)],
                           self.parameters['channel_range' + str(i)],
                           self.parameters['impedance' + str(i)])
            if self.parameters.get('bwlimit' + str(i), None) is not None:
                self._call_dll('AlazarSetBWLimit',
                               self._handle, 2**(i-1),
                               self.parameters['bwlimit' + str(i)])

        self._call_dll('AlazarSetTriggerOperation',
                       self._handle, self.trigger_operation,
                       self.trigger_engine1, self.trigger_source1,
                       self.trigger_slope1, self.trigger_level1,
                       self.trigger_engine2, self.trigger_source2,
                       self.trigger_slope2, self.trigger_level2)

        self._call_dll('AlazarSetExternalTrigger',
                       self._handle, self.external_trigger_coupling,
                       self.external_trigger_range)

        self._call_dll('AlazarSetTriggerDelay',
                       self._handle, self.trigger_delay)

        self._call_dll('AlazarSetTriggerTimeOut',
                       self._handle, self.timeout_ticks)

        self._call_dll('AlazarConfigureAuxIO',
                       self._handle, self.aux_io_mode,
                       self.aux_io_param)
        self._parameters_synced = True

    def _get_channel_info(self, handle: int) -> Tuple[int,int]:
        bps = ctypes.c_uint8(0)  # bps bits per sample
        max_s = ctypes.c_uint32(0)  # max_s memory size in samples
        self._call_dll('AlazarGetChannelInfo',
                       handle,
                       ctypes.byref(max_s),
                       ctypes.byref(bps))
        return max_s.value, bps.value

    def acquire(self, mode=None, samples_per_record=None,
                records_per_buffer=None, buffers_per_acquisition=None,
                channel_selection=None, transfer_offset=None,
                external_startcapture=None, enable_record_headers=None,
                alloc_buffers=None, fifo_only_streaming=None,
                interleave_samples=None, get_processed_data=None,
                allocated_buffers=None, buffer_timeout=None,
                acquisition_controller=None):
        """
        perform a single acquisition with the Alazar board, and set certain
        parameters to the appropriate values
        for the parameters, see the ATS-SDK programmer's guide

        Args:
            mode:
            samples_per_record:
            records_per_buffer:
            buffers_per_acquisition:
            channel_selection:
            transfer_offset:
            external_startcapture:
            enable_record_headers:
            alloc_buffers:
            fifo_only_streaming:
            interleave_samples:
            get_processed_data:
            allocated_buffers:
            buffer_timeout:
            acquisition_controller: An instance of an acquisition controller
                that handles the dataflow of an acquisition

        Returns:
            Whatever is given by acquisition_controller.post_acquire method
        """
        # region set parameters from args
        if self._parameters_synced == False:
            raise RuntimeError("You must sync parameters to Alazar card "
                               "before calling acquire by calling "
                               "sync_parameters_to_card")
        self._set_if_present('mode', mode)
        self._set_if_present('samples_per_record', samples_per_record)
        self._set_if_present('records_per_buffer', records_per_buffer)
        self._set_if_present('buffers_per_acquisition',
                             buffers_per_acquisition)
        self._set_if_present('channel_selection', channel_selection)
        self._set_if_present('transfer_offset', transfer_offset)
        self._set_if_present('external_startcapture', external_startcapture)
        self._set_if_present('enable_record_headers', enable_record_headers)
        self._set_if_present('alloc_buffers', alloc_buffers)
        self._set_if_present('fifo_only_streaming', fifo_only_streaming)
        self._set_if_present('interleave_samples', interleave_samples)
        self._set_if_present('get_processed_data', get_processed_data)
        self._set_if_present('allocated_buffers', allocated_buffers)
        self._set_if_present('buffer_timeout', buffer_timeout)

        # endregion
        mode = self.mode.get()
        if mode not in ('TS', 'NPT'):
            raise Exception("Only the 'TS' and 'NPT' modes are implemented "
                            "at this point")

        # -----set final configurations-----

        # Abort any previous measurement
        self._call_dll('AlazarAbortAsyncRead', self._handle)

        buffers_per_acquisition = self.buffers_per_acquisition.raw_value
        samples_per_record = self.samples_per_record.raw_value
        records_per_buffer = self.records_per_buffer.raw_value
        # Set record size for NPT mode
        if mode == 'NPT':
            pretriggersize = 0  # pretriggersize is 0 for NPT always
            post_trigger_size = samples_per_record
            self._call_dll('AlazarSetRecordSize',
                           self._handle, pretriggersize,
                           post_trigger_size)

        # set acquisition parameters here for NPT, TS mode
        samples_per_buffer = 0

        acquire_flags = (self.mode.raw_value |
                         self.external_startcapture.raw_value |
                         self.enable_record_headers.raw_value |
                         self.alloc_buffers.raw_value |
                         self.fifo_only_streaming.raw_value |
                         self.interleave_samples.raw_value |
                         self.get_processed_data.raw_value)

        if mode == 'NPT':
            records_per_acquisition = (
                records_per_buffer * buffers_per_acquisition)
            self._call_dll('AlazarBeforeAsyncRead',
                           self._handle, self.channel_selection.raw_value,
                           self.transfer_offset.raw_value,
                           samples_per_record,
                           records_per_buffer, records_per_acquisition,
                           acquire_flags)

        elif mode == 'TS':
            if (samples_per_record % buffers_per_acquisition != 0):
                logger.warning('buffers_per_acquisition is not a divisor of '
                                'samples per record which it should be in '
                                'Ts mode, rounding down in samples per buffer '
                                'calculation')
            samples_per_buffer = int(samples_per_record /
                                     buffers_per_acquisition)
            if self.records_per_buffer.raw_value != 1:
                logger.warning('records_per_buffer should be 1 in TS mode, '
                                'defauling to 1')
                self.records_per_buffer.set(1)
            records_per_buffer = self.records_per_buffer.raw_value

            self._call_dll('AlazarBeforeAsyncRead',
                           self._handle, self.channel_selection.raw_value,
                           self.transfer_offset.raw_value, samples_per_buffer,
                           records_per_buffer, buffers_per_acquisition,
                           acquire_flags)

        # bytes per sample
        max_s, bps = self._get_channel_info(self._handle)
        # TODO(JHN) Why +7 I guess its to do ceil division?
        bytes_per_sample = (bps + 7) // 8
        # bytes per record
        bytes_per_record = bytes_per_sample * samples_per_record

        # channels
        channels_binrep = self.channel_selection.raw_value
        number_of_channels = self.get_num_channels(channels_binrep)

        # bytes per sample
        max_s, bps = self._get_channel_info(self._handle)
        # TODO(JHN) Why +7 I guess its to do ceil division?
        bytes_per_sample = (bps + 7) // 8
        # bytes per record
        bytes_per_record = bytes_per_sample * samples_per_record

        # bytes per buffer
        bytes_per_buffer = (bytes_per_record *
                            records_per_buffer * number_of_channels)

        sample_type = ctypes.c_uint8
        if bytes_per_sample > 1:
            sample_type = ctypes.c_uint16

        self.clear_buffers()

        # make sure that allocated_buffers <= buffers_per_acquisition
        allocated_buffers = self.allocated_buffers.raw_value
        buffers_per_acquisition = self.buffers_per_acquisition.raw_value

        if allocated_buffers > buffers_per_acquisition:
            logger.warning("'allocated_buffers' should be <= "
                            "'buffers_per_acquisition'. Defaulting 'allocated_buffers'"
                            f" to {buffers_per_acquisition}")
            self.allocated_buffers.set(buffers_per_acquisition)

        allocated_buffers = self.allocated_buffers.raw_value
        buffer_recycling = buffers_per_acquisition > allocated_buffers
        for k in range(allocated_buffers):
            try:
                self.buffer_list.append(Buffer(sample_type, bytes_per_buffer))
            except:
                self.clear_buffers()
                raise

        # post buffers to Alazar
        try:
            for buf in self.buffer_list:
                self._call_dll('AlazarPostAsyncBuffer',
                               self._handle, ctypes.cast(buf.addr, ctypes.c_void_p), buf.size_bytes)

            # -----start capture here-----
            acquisition_controller.pre_start_capture()
            start = time.clock()  # Keep track of when acquisition started
            # call the startcapture method
            self._call_dll('AlazarStartCapture', self._handle)

            acquisition_controller.pre_acquire()

            # buffer handling from acquisition
            buffers_completed = 0
            bytes_transferred = 0
            buffer_timeout = self.buffer_timeout.raw_value

            done_setup = time.clock()
            while (buffers_completed < self.buffers_per_acquisition.get()):
                # Wait for the buffer at the head of the list of available
                # buffers to be filled by the board.
                buf = self.buffer_list[buffers_completed % allocated_buffers]
                self._call_dll('AlazarWaitAsyncBufferComplete',
                               self._handle, ctypes.cast(buf.addr, ctypes.c_void_p), buffer_timeout)

                # TODO(damazter) (C) last series of buffers must be handled
                # exceptionally
                # (and I want to test the difference) by changing buffer
                # recycling for the last series of buffers

                # if buffers must be recycled, extract data and repost them
                # otherwise continue to next buffer
                if buffer_recycling:
                    acquisition_controller.handle_buffer(buf.buffer, buffers_completed)
                    self._call_dll('AlazarPostAsyncBuffer',
                                   self._handle, ctypes.cast(buf.addr, ctypes.c_void_p), buf.size_bytes)
                buffers_completed += 1
                bytes_transferred += buf.size_bytes
        finally:
            # stop measurement here
            done_capture = time.clock()
            self._call_dll('AlazarAbortAsyncRead', self._handle)
        time_done_abort = time.clock()
        # -----cleanup here-----
        # extract data if not yet done
        if not buffer_recycling:
            for i, buf in enumerate(self.buffer_list):
                acquisition_controller.handle_buffer(buf.buffer, i)
        time_done_handling = time.clock()
        # free up memory
        self.clear_buffers()

        time_done_free_mem = time.clock()
        # check if all parameters are up to date
        # Getting IDN is very slow so skip that
        for name, p in self.parameters.items():
            if isinstance(p, TraceParameter):
                if p.synced_to_card == False:
                    raise RuntimeError(f"TraceParameter {p} not synced to "
                                       f"Alazar card detected. Aborting. Data "
                                       f"may be corrupt")


        # Compute the total transfer time, and display performance information.
        end_time = time.clock()
        transfer_time_sec = end_time - start
        setup_time = done_setup - start
        capture_time = done_capture - done_setup
        abort_time = time_done_abort - done_capture
        handling_time = time_done_handling - time_done_abort
        free_mem_time = time_done_free_mem - time_done_handling
        buffers_per_sec = 0
        bytes_per_sec = 0
        records_per_sec = 0
        if transfer_time_sec > 0:
            buffers_per_sec = buffers_completed / transfer_time_sec
            bytes_per_sec = bytes_transferred / transfer_time_sec
            records_per_sec = (records_per_buffer *
                               buffers_completed / transfer_time_sec)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Captured %d buffers (%f buffers per sec)" %
                         (buffers_completed, buffers_per_sec))
            logger.debug("Captured %d records (%f records per sec)" %
                         (records_per_buffer * buffers_completed, records_per_sec))
            logger.debug("Transferred {:g} bytes ({:g} "
                         "bytes per sec)".format(bytes_transferred, bytes_per_sec))
            logger.debug("Pre capture setup took {}".format(setup_time))
            logger.debug("Capture took {}".format(capture_time))
            logger.debug("abort took {}".format(abort_time))
            logger.debug("handling took {}".format(handling_time))
            logger.debug("free mem took {}".format(free_mem_time))
        # return result
        return acquisition_controller.post_acquire()

    def _set_if_present(self, param_name: str, value: Union[int,str,float]) -> None:
        if value is not None:
            parameter = self.parameters[param_name]
            parameter.set(value)

    def _set_list_if_present(self, param_base: str, value: Sequence[Union[int,str,float]]) -> None:
        if value is not None:
            for i, v in enumerate(value):
                parameter = self.parameters[param_base + str(i + 1)]
                parameter.set(v)

    def _call_dll(self, func_name: str, *args) -> None:
        """
        Execute a dll function `func_name`, passing it the given arguments

        For each argument in the list
        - If an arg is a TraceParameter of this instrument, the parameter
          value from `.raw_value` is used. If the call succeeds, these
          parameters will be marked as updated using their `._set_updated()`
          method
        - If a regular parameter the raw_value is used and uptodate is tracked
          outside this function
        - Otherwise the arg is used directly
        """
        # create the argument list
        args_out: List[int] = []
        update_params: List[Parameter] = []
        for arg in args:
            if isinstance(arg, Parameter):
                args_out.append(arg.raw_value)
                update_params.append(arg)
            else:
                args_out.append(arg)
        # may be useful to log this but is called a lot so leave it out for now
        # logger.debug("calling dll func {} with args: \n {}".format(func_name, args_out))
        # run the function
        func = getattr(self._ATS_dll, func_name)
        try:
            return_code = func(*args_out)
        except Exception as e:
            logger.exception("Exception in DLL call")
            raise e

        # check for errors
        if (return_code != self._success) and (return_code != 518):
            # TODO(damazter) (C) log error

            argrepr = repr(args_out)
            if len(argrepr) > 100:
                argrepr = argrepr[:96] + '...]'

            if return_code not in self._error_codes:
                raise RuntimeError(
                    'unknown error {} from function {} with args: {}'.format(
                        return_code, func_name, argrepr))
            raise RuntimeError(
                'error {}: {} from function {} with args: {}'.format(
                    return_code, self._error_codes[return_code], func_name,
                    argrepr))

        # mark parameters updated (only after we've checked for errors)
        for param in update_params:
            if isinstance(param, TraceParameter):
                param._set_updated()


    def clear_buffers(self) -> None:
        """
        This method uncommits all buffers that were committed by the driver.
        This method only has to be called when the acquistion crashes, otherwise
        the driver will uncommit the buffers itself

        Returns:
            None

        """
        for b in self.buffer_list:
            b.free_mem()
        logger.debug("buffers cleared")
        self.buffer_list = []

    def signal_to_volt(self, channel: int, signal: int) -> float:
        """
        convert a value from a buffer to an actual value in volts based on the
        ranges of the channel

        Args:
            channel: number of the channel where the signal value came from
            signal: the value that needs to be converted

        Returns:
             the corresponding value in volts
        """
        # TODO(damazter) (S) check this
        # TODO(damazter) (M) use byte value if range{channel}
        return (((signal - 127.5) / 127.5) *
                (self.parameters['channel_range' + str(channel)].get()))

    def get_sample_rate(self, include_decimation: bool=True) -> Union[float,int]:
        """
        Obtain the effective sampling rate of the acquisition
        based on clock speed and decimation

        Returns:
            the number of samples (per channel) per second
        """
        if (self.clock_source.get() == 'EXTERNAL_CLOCK_10MHz_REF'
            and 'external_sample_rate' in self.parameters):
            rate = self.external_sample_rate.get()
            # if we are using an external ref clock the sample rate
            # is set as an integer and not value mapped so we use a different
            # parameter to represent it
        elif self.sample_rate.get() == 'EXTERNAL_CLOCK':
            raise Exception('External clock is used, alazar driver '
                            'could not determine sample speed.')
        else:
            rate = self.sample_rate.get()
        if rate == '1GHz_REFERENCE_CLOCK':
            rate = 1e9

        if include_decimation:
            decimation = self.decimation.get()
        else:
            decimation = 0
        if decimation > 0:
            return rate / decimation
        else:
            return rate

    @staticmethod
    def get_num_channels(byte_rep: int) -> int:
        """
        Return the number of channels for a specific channel mask


        Each single channel is represented by a bitarray with one
        non zero entry i.e. powers of two. All multichannel masks can be
        constructed by summing the single channel ones. However, not all
        configurations are supported. See table 4 Input Channel Configurations
        on page 241 of the Alazar SDK manual. This contains the complete
        mapping for all current Alazar cards. It's left to the driver to
        ensure that only the ones supported for a specific card can be
        selected
        """
        one_channels = tuple(2**i for i in range(16))
        two_channels = (3, 5, 6, 9, 10, 12)
        four_channels = (255,)
        sixteen_channels = (65535,)

        if byte_rep in one_channels:
            return 1
        elif byte_rep in two_channels:
            return 2
        elif byte_rep in four_channels:
            return 4
        elif byte_rep in sixteen_channels:
            return 16
        else:
            raise RuntimeError('Invalid channel configuration supplied')


class Buffer:
    """Buffer suitable for DMA transfers.

    AlazarTech digitizers use direct memory access (DMA) to transfer
    data from digitizers to the computer's main memory. This class
    abstracts a memory buffer on the host, and ensures that all the
    requirements for DMA transfers are met.

    Buffer export a 'buffer' member, which is a NumPy array view
    of the underlying memory buffer

    Args:
        c_sample_type (ctypes type): The datatype of the buffer to create.
        size_bytes (int): The size of the buffer to allocate, in bytes.
    """
    def __init__(self, c_sample_type, size_bytes):
        self.size_bytes = size_bytes

        npSampleType = {
            ctypes.c_uint8: np.uint8,
            ctypes.c_uint16: np.uint16,
            ctypes.c_uint32: np.uint32,
            ctypes.c_int32: np.int32,
            ctypes.c_float: np.float32
        }.get(c_sample_type, 0)

        bytes_per_sample = {
            ctypes.c_uint8:  1,
            ctypes.c_uint16: 2,
            ctypes.c_uint32: 4,
            ctypes.c_int32:  4,
            ctypes.c_float:  4
        }.get(c_sample_type, 0)

        self._allocated = True
        self.addr = None
        if os.name == 'nt':
            MEM_COMMIT = 0x1000
            PAGE_READWRITE = 0x4
            ctypes.windll.kernel32.VirtualAlloc.restype = ctypes.c_void_p
            self.addr = ctypes.windll.kernel32.VirtualAlloc(
                0, ctypes.c_long(size_bytes), MEM_COMMIT, PAGE_READWRITE)
        else:
            self._allocated = False
            raise Exception("Unsupported OS")

        ctypes_array = (c_sample_type *
                        (size_bytes // bytes_per_sample)).from_address(self.addr)
        self.buffer = np.frombuffer(ctypes_array, dtype=npSampleType)
        self.ctypes_buffer = ctypes_array
        pointer, read_only_flag = self.buffer.__array_interface__['data']

    def free_mem(self):
        """
        uncommit memory allocated with this buffer object
        """

        self._allocated = False
        if os.name == 'nt':
            MEM_RELEASE = 0x8000
            ctypes.windll.kernel32.VirtualFree.restype = ctypes.c_int
            ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(self.addr), 0, MEM_RELEASE)
        else:
            self._allocated = True
            raise Exception("Unsupported OS")

    def __del__(self):
        """
        If python garbage collects this object, __del__ should be called and it
        is the last chance to uncommit the memory to prevent a memory leak.
        This method is not very reliable so users should not rely on this
        functionality
        """
        if self._allocated:
            self.free_mem()
            logger.warning(
                'Buffer prevented memory leak; Memory released to Windows.\n'
                'Memory should have been released before buffer was deleted.')


class AcquisitionController(Instrument):
    """
    This class represents all choices that the end-user has to make regarding
    the data-acquisition. this class should be subclassed to program these
    choices.

    The basic structure of an acquisition is:

        - Call to AlazarTech_ATS.acquire internal configuration
        - Call to acquisitioncontroller.pre_start_capture
        - Call to the start capture of the Alazar board
        - Call to acquisitioncontroller.pre_acquire
        - Loop over all buffers that need to be acquired
          dump each buffer to acquisitioncontroller.handle_buffer
          (only if buffers need to be recycled to finish the acquisiton)
        - Dump remaining buffers to acquisitioncontroller.handle_buffer
          alazar internals
        - Return acquisitioncontroller.post_acquire

    Attributes:
        _alazar: a reference to the alazar instrument driver
    """
    def __init__(self, name, alazar_name, **kwargs):
        """
        Args:
            alazar_name: The name of the alazar instrument on the server
        """
        super().__init__(name, **kwargs)
        self._alazar = self.find_instrument(alazar_name,
                                            instrument_class=AlazarTech_ATS)

    def _get_alazar(self):
        """
        returns a reference to the alazar instrument. A call to self._alazar is
        quicker, so use that if in need for speed
        :return: reference to the Alazar instrument
        """
        return self._alazar

    def pre_start_capture(self):
        """
        Use this method to prepare yourself for the data acquisition
        The Alazar instrument will call this method right before
        'AlazarStartCapture' is called
        """
        raise NotImplementedError(
            'This method should be implemented in a subclass')

    def pre_acquire(self):
        """
        This method is called immediately after 'AlazarStartCapture' is called
        """
        raise NotImplementedError(
            'This method should be implemented in a subclass')

    def handle_buffer(self, buffer, buffer_number=None):
        """
        This method should store or process the information that is contained
        in the buffers obtained during the acquisition.

        Args:
            buffer: np.array with the data from the Alazar card
            buffer_number: counter for which buffer we are handling

        Returns:
            something, it is ignored in any case
        """
        raise NotImplementedError(
            'This method should be implemented in a subclass')

    def post_acquire(self):
        """
        This method should return any information you want to save from this
        acquisition. The acquisition method from the Alazar driver will use
        this data as its own return value

        Returns:
            this function should return all relevant data that you want
            to get form the acquisition
        """
        raise NotImplementedError(
            'This method should be implemented in a subclass')
