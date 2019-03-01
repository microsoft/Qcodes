import ctypes
import logging
import numpy as np
import time
import os
import warnings
import sys
import asyncio
import concurrent

from threading import Lock
from functools import partial
from typing import List, Dict, Union, Tuple, Sequence, Awaitable, Optional
from contextlib import contextmanager

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
from qcodes.instrument_drivers.AlazarTech.ats_api import AlazarATSAPI, BOARD_NAMES
from qcodes.utils.async_utils import sync
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

    api: AlazarATSAPI

    # override channels in a subclass if needed
    channels = 2

    @classmethod
    async def find_boards_async(cls, dll_path: str = None) -> List[dict]:
        api = AlazarATSAPI(dll_path or cls.dll_path)

        system_count = await api.num_of_systems_async()
        boards = []
        for system_id in range(1, system_count + 1):
            board_count = await api.boards_in_system_by_system_id_async(system_id)
            for board_id in range(1, board_count + 1):
                boards.append(await cls.get_board_info_async(api, system_id, board_id))
        return boards

    @classmethod
    def find_boards(cls, dll_path: str=None) -> List[dict]:
        """
        Find Alazar boards connected

        Args:
            dll_path: (string) path of the Alazar driver dll

        Returns:
            list: list of board info for each connected board
        """
        return sync(cls.find_boards_async(dll_path))

    # TODO(nataliejpg) this needs fixing..., dll can't be a string
    @classmethod
    def get_board_info(cls, api: AlazarATSAPI, system_id: int,
                       board_id: int) -> Dict[str, Union[str, int]]:
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
        return sync(cls.get_board_info_async(api, system_id, board_id))

    @classmethod
    async def get_board_info_async(cls, api: AlazarATSAPI, system_id: int,
                                   board_id: int) -> Dict[str, Union[str, int]]:
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
        board_kind = api._board_names[api.get_board_kind(handle)]

        max_s, bps = await board._get_channel_info_async(handle)
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
        self.api = AlazarATSAPI(dll_path or self.dll_path)

        self._parameters_synced = False
        self._handle = self.api.get_board_by_system_id(system_id, board_id)

        if not self._handle:
            raise Exception('AlazarTech_ATS not found at '
                            'system {}, board {}'.format(system_id, board_id))

        self.buffer_list: List['Buffer'] = []

    def query_capability(self, capability: int) -> int:
        value = ctypes.c_uint32(0)
        self.api.query_capability(
            self._handle, capability, 0, ctypes.byref(value)
        )
        return value.value

    def get_idn(self) -> dict:
        # TODO this is really Dict[str, Optional[Union[str,int]]]
        # But that is inconsistent with the super class. We should consider
        # if ints and floats are allowed as values in the dict
        """
        This methods gets the most relevant information of this instrument

        The firmware version reported should match the version number of
        downloadable fw files from AlazarTech. But note that the firmware
        version has often been found to be incorrect for several firmware
        versions. At the time of writing it is known to be correct for the
        9360 (v 21.07) and 9373 (v 30.04) but incorrect for several earlier
        versions. In Alazar DSO this is reported as FPGA Version.

        Returns:

            Dictionary containing

                - 'firmware': as string
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
        board_kind = BOARD_NAMES[self.api.get_board_kind(self._handle)]
        max_s, bps = self._get_channel_info(self._handle)

        major = ctypes.c_uint8(0)
        minor = ctypes.c_uint8(0)
        revision = ctypes.c_uint8(0)
        self.api.get_cpld_version(
            self._handle,
            ctypes.byref(major),
            ctypes.byref(minor)
        )
        cpld_ver = str(major.value) + "." + str(minor.value)

        self.api.get_driver_version(
            ctypes.byref(major),
            ctypes.byref(minor),
            ctypes.byref(revision)
        )
        driver_ver = str(major.value)+"."+str(minor.value) + \
            "."+str(revision.value)

        self.api.get_sdk_version(
            ctypes.byref(major),
            ctypes.byref(minor),
            ctypes.byref(revision)
        )
        sdk_ver = str(major.value)+"."+str(minor.value)+"."+str(revision.value)

        serial = str(self.query_capability(0x10000024))
        capability_str = str(self.query_capability(0x10000026))
        latest_cal_date = (capability_str[0:2] + "-" +
                           capability_str[2:4] + "-" +
                           capability_str[4:6])

        memory_size = str(self.query_capability(0x1000002A))
        asopc_type = self.query_capability(0x1000002C)

        # see the ATS-SDK programmer's guide
        # about the encoding of the link speed
        pcie_link_speed = str(self.query_capability(
            0x10000030) * 2.5 / 10) + "GB/s"
        pcie_link_width = str(self.query_capability(0x10000031))

        # Alazartech has confirmed in a support mail that this
        # is the way to get the firmware version
        firmware_major = (asopc_type >> 16) & 0xff
        firmware_minor = (asopc_type >> 24) & 0xf
        # firmware_minor above does not contain any prefixed zeros
        # but the minor version is always 2 digits.
        firmware_version = f'{firmware_major}.{firmware_minor:02d}'

        return {'firmware': firmware_version,
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
            raise RuntimeError(
                "Both sample_rate and external_sample_rate supplied")

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
                raise RuntimeError(
                    "Using Internal clock but parameter sample_rate is not set")
            if self.external_sample_rate.raw_value != 'UNDEFINED':
                warnings.warn("Using Internal clock but parameter external_sample_rate is set."
                              "This will have no effect and is ignored")
            # mark the unused parameter as up to date
            self.external_sample_rate._set_updated()
            sample_rate = self.sample_rate

        self.api.set_capture_clock(
            self._handle, self.clock_source, sample_rate,
            self.clock_edge, self.decimation
        )

        for i in range(1, self.channels + 1):
            self.api.input_control(
                self._handle, 2**(i-1),
                self.parameters['coupling' + str(i)],
                self.parameters['channel_range' + str(i)],
                self.parameters['impedance' + str(i)]
            )
            if self.parameters.get('bwlimit' + str(i), None) is not None:
                self.api.set_bw_limit(
                    self._handle, 2**(i-1),
                    self.parameters['bwlimit' + str(i)]
                )

        self.api.set_trigger_operation(
            self._handle, self.trigger_operation,
            self.trigger_engine1, self.trigger_source1,
            self.trigger_slope1, self.trigger_level1,
            self.trigger_engine2, self.trigger_source2,
            self.trigger_slope2, self.trigger_level2
        )
        self.api.set_external_trigger(
            self._handle, self.external_trigger_coupling,
            self.external_trigger_range
        )
        self.api.set_trigger_delay(self._handle, self.trigger_delay)
        self.api.set_trigger_time_out(self._handle, self.timeout_ticks)
        self.api.configure_aux_io(
            self._handle, self.aux_io_mode,
            self.aux_io_param
        )
        self._parameters_synced = True

    def _get_channel_info(self, handle: int) -> Tuple[int, int]:
        bps = ctypes.c_uint8(0)  # bps bits per sample
        max_s = ctypes.c_uint32(0)  # max_s memory size in samples
        self.api.get_channel_info(
            handle,
            ctypes.byref(max_s),
            ctypes.byref(bps)
        )
        return max_s.value, bps.value

    async def _get_channel_info_async(self, handle: int) -> Tuple[int, int]:
        bps = ctypes.c_uint8(0)  # bps bits per sample
        max_s = ctypes.c_uint32(0)  # max_s memory size in samples
        await self.api.get_channel_info_async(
            handle,
            ctypes.byref(max_s),
            ctypes.byref(bps)
        )
        return max_s.value, bps.value

    async def allocate_and_post_buffer(self, sample_type, n_bytes) -> "Buffer":
        buffer = Buffer(sample_type, n_bytes)
        await self.api.post_async_buffer_async(
            self._handle, ctypes.cast(
                buffer.addr, ctypes.c_void_p), buffer.size_bytes
        )
        return buffer

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
        loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_acquire(
            mode=mode,
            samples_per_record=samples_per_record,
            records_per_buffer=records_per_buffer,
            buffers_per_acquisition=buffers_per_acquisition,
            channel_selection=channel_selection,
            transfer_offset=transfer_offset,
            external_startcapture=external_startcapture,
            enable_record_headers=enable_record_headers,
            alloc_buffers=alloc_buffers,
            fifo_only_streaming=fifo_only_streaming,
            interleave_samples=interleave_samples,
            get_processed_data=get_processed_data,
            allocated_buffers=alloc_buffers,
            buffer_timeout=buffer_timeout,
            acquisition_controller=acquisition_controller
        ))

    async def async_acquire(self, mode=None, samples_per_record=None,
                            records_per_buffer=None, buffers_per_acquisition=None,
                            channel_selection=None, transfer_offset=None,
                            external_startcapture=None, enable_record_headers=None,
                            alloc_buffers=None, fifo_only_streaming=None,
                            interleave_samples=None, get_processed_data=None,
                            allocated_buffers=None, buffer_timeout=None,
                            acquisition_controller=None):

        # region set parameters from args
        start_func = time.perf_counter()
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

        buffers_per_acquisition = self.buffers_per_acquisition.raw_value
        samples_per_record = self.samples_per_record.raw_value
        records_per_buffer = self.records_per_buffer.raw_value

        # Set record size for NPT mode
        if mode == 'NPT':
            pretriggersize = 0  # pretriggersize is 0 for NPT always
            post_trigger_size = samples_per_record
            await self.api.set_record_size_async(
                self._handle, pretriggersize,
                post_trigger_size
            )
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
            await self.api.before_async_read_async(
                self._handle, self.channel_selection.raw_value,
                self.transfer_offset.raw_value,
                samples_per_record,
                records_per_buffer, records_per_acquisition,
                acquire_flags
            )
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

            await self.api.before_async_read_async(
                self._handle, self.channel_selection.raw_value,
                self.transfer_offset.raw_value, samples_per_buffer,
                records_per_buffer, buffers_per_acquisition,
                acquire_flags
            )

        # bytes per sample
        _, bps = await self._get_channel_info_async(self._handle)
        # TODO(JHN) Why +7 I guess its to do ceil division?
        bytes_per_sample = (bps + 7) // 8
        # bytes per record
        bytes_per_record = bytes_per_sample * samples_per_record

        # channels
        channels_binrep = self.channel_selection.raw_value
        number_of_channels = self.get_num_channels(channels_binrep)

        # bytes per buffer
        bytes_per_buffer = (bytes_per_record *
                            records_per_buffer * number_of_channels)

        sample_type = ctypes.c_uint16 if bytes_per_sample > 1 else ctypes.c_uint8

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

        # post buffers to Alazar
        try:
            # NB: gather does not ensure that the tasks complete in the same order,
            #     but it does guarantee that the sequence of results is in the same
            #     order as the sequence of tasks, irrespective of when each task
            #     completes.
            #     We use this to make sure that we can begin allocating the next
            #     buffer while each buffer is being posted.
            for buf in await asyncio.gather(*(
                self.allocate_and_post_buffer(sample_type, bytes_per_buffer)
                for _ in range(allocated_buffers)
            )):
                self.buffer_list.append(buf)

            # -----start capture here-----
            acquisition_controller.pre_start_capture()
            start = time.perf_counter()  # Keep track of when acquisition started
            # call the startcapture method
            await self.api.start_capture_async(self._handle)
            acquisition_controller.pre_acquire()

            # buffer handling from acquisition
            buffers_completed = 0
            bytes_transferred = 0
            buffer_timeout = self.buffer_timeout.raw_value

            done_setup = time.perf_counter()
            print(f"[{time.time()}] Entering buffer wait loop.")
            while (buffers_completed < self.buffers_per_acquisition.get()):
                # Wait for the buffer at the head of the list of available
                # buffers to be filled by the board.
                buf = self.buffer_list[buffers_completed % allocated_buffers]
                await self.api.wait_async_buffer_complete_async(
                    self._handle, ctypes.cast(
                        buf.addr, ctypes.c_void_p), buffer_timeout
                )

                acquisition_controller.buffer_done_callback(buffers_completed)

                # if buffers must be recycled, extract data and repost them
                # otherwise continue to next buffer
                if buffer_recycling:
                    acquisition_controller.handle_buffer(
                        buf.buffer, buffers_completed)
                    await self.api.post_async_buffer_async(
                        self._handle, ctypes.cast(
                            buf.addr, ctypes.c_void_p), buf.size_bytes
                    )
                buffers_completed += 1
                bytes_transferred += buf.size_bytes
            print(f"[{time.time()}] Exited buffer wait loop.")
        finally:
            # stop measurement here
            done_capture = time.perf_counter()
            await self.api.abort_async_read_async(self._handle)

        # If we made it here, we should have completed all buffers.
        assert buffers_completed == self.buffers_per_acquisition.get()
        time_done_abort = time.perf_counter()
        # -----cleanup here-----
        # extract data if not yet done
        if not buffer_recycling:
            for i, buf in enumerate(self.buffer_list):
                acquisition_controller.handle_buffer(buf.buffer, i)
        time_done_handling = time.perf_counter()
        # free up memory
        self.clear_buffers()

        time_done_free_mem = time.perf_counter()
        # check if all parameters are up to date
        # Getting IDN is very slow so skip that
        for _, p in self.parameters.items():
            if isinstance(p, TraceParameter):
                if p.synced_to_card == False:
                    raise RuntimeError(f"TraceParameter {p} not synced to "
                                       f"Alazar card detected. Aborting. Data "
                                       f"may be corrupt")

        # Compute the total transfer time, and display performance information.
        end_time = time.perf_counter()
        tot_time = end_time - start_func
        transfer_time_sec = end_time - start
        presetup_time = start - start_func
        setup_time = done_setup - start
        capture_time = done_capture - done_setup
        abort_time = time_done_abort - done_capture
        handling_time = time_done_handling - time_done_abort
        free_mem_time = time_done_free_mem - time_done_handling
        buffers_per_sec: float = 0
        bytes_per_sec: float = 0
        records_per_sec: float = 0
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
            logger.debug("Pre setup took {}".format(presetup_time))
            logger.debug("Pre capture setup took {}".format(setup_time))
            logger.debug("Capture took {}".format(capture_time))
            logger.debug("abort took {}".format(abort_time))
            logger.debug("handling took {}".format(handling_time))
            logger.debug("free mem took {}".format(free_mem_time))
            logger.debug("tot acquire time is {}".format(tot_time))

        # return result
        return acquisition_controller.post_acquire()

    async_acquire.__doc__ = acquire.__doc__

    def _set_if_present(self, param_name: str, value: Union[int, str, float]) -> None:
        if value is not None:
            parameter = self.parameters[param_name]
            parameter.set(value)

    def _set_list_if_present(self, param_base: str, value: Sequence[Union[int, str, float]]) -> None:
        if value is not None:
            for i, v in enumerate(value):
                parameter = self.parameters[param_base + str(i + 1)]
                parameter.set(v)

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

    def get_sample_rate(self, include_decimation: bool=True) -> Union[float, int]:
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

    def _read_register(self, offset: int) -> int:
        """
        Read a value from a given register in the Alazars memory

        Args:
            offset: Offset into he memmory to read from

        Returns:
            The value read as en integer
        """
        output = ctypes.c_uint32(0)
        pwd = ctypes.c_uint32(0x32145876)
        self.api.read_register(
            self._handle,
            offset,
            ctypes.byref(output),
            pwd
        )
        return output.value

    def _write_register(self, offset: int, value: int) -> None:
        """
        Write a value to a given offset in the Alazars memory

        Args:
            offset: The offset to write to
            value: The value to write
        """
        pwd = ctypes.c_uint32(0x32145876)
        self.api.write_register(
            self._handle,
            offset,
            value,
            pwd
        )


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

    def free_mem(self):
        """
        uncommit memory allocated with this buffer object
        """

        self._allocated = False
        if os.name == 'nt':
            MEM_RELEASE = 0x8000
            ctypes.windll.kernel32.VirtualFree.restype = ctypes.c_int
            ctypes.windll.kernel32.VirtualFree(
                ctypes.c_void_p(self.addr), 0, MEM_RELEASE)
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

    async def handle_buffer_async(self, buffer, buffer_number=None):
        """
        This method should store or process the information that is contained
        in the buffers obtained during the acquisition.

        Args:
            buffer: np.array with the data from the Alazar card
            buffer_number: counter for which buffer we are handling

        Returns:
            something, it is ignored in any case
        """
        return self.handle_buffer(buffer, buffer_number=buffer_number)

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

    def buffer_done_callback(self, buffers_completed):
        """
        This method is called when a buffer is completed. It can be used
        if you want to implement an event that happens for each buffer.
        You will probably want to combine this with `AUX_IN_TRIGGER_ENABLE` to wait
        before starting capture of the next buffer.

        Args:
            buffers_completed: how many buffers have been completed and copied
            to local memory at the time of this callback.
        """
        pass
