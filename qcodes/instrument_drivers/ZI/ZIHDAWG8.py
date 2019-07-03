import csv
import json
import os
import re
import textwrap
import time
from functools import partial
from typing import Any, List, Tuple, Union, Sequence, Dict, Optional

import zhinst.utils

from qcodes import Instrument
from qcodes.utils import validators as validators

WARNING_CLIPPING = r"^Warning \(line: [0-9]+\): [a-zA-Z0-9_]+ has a higher " \
                   r"amplitude than 1.0, waveform amplitude will be limited " \
                   r"to 1.0$"
WARNING_TRUNCATING = r"^Warning \(line: [0-9]+\): waveform \S+ cut " \
                     r"down to playable length from [0-9]+ to [0-9]+ samples " \
                     r"\(should be a multiple of 8 samples for single channel" \
                     r" or 4 samples for dual channel waveforms\)$"
WARNING_ANY = r"^Warning \(line: [0-9]+\):.*$"


class CompilerError(Exception):
    """ Errors that occur during compilation of sequence programs."""


class ZIHDAWG8(Instrument):
    """
    QCoDeS driver for ZI HDAWG8.
    Requires ZI LabOne software to be installed on the computer running QCoDeS
    (tested using LabOne 18.05.54618 and firmware 53866).
    Furthermore, the Data Server and Web Server must be running and a connection
    between the two must be made.

    Compiler warnings, when uploading and compiling a sequence program, can
    be treated as errors. This is desirable if the user of the driver does
    not want clipping or truncating of waveform to happen silently by the
    compiler. Warnings are constants on the module level and can be added to the
    drivers attribute ``warnings_as_errors``. If warning are added, they
    will raise a CompilerError.
    """

    def __init__(self, name: str, device_id: str, **kwargs) -> None:
        """
        Create an instance of the instrument.

        Args:
            name: The internal QCoDeS name of the instrument
            device_ID: The device name as listed in the web server.
        """
        super().__init__(name, **kwargs)
        self.api_level = 6
        (self.daq, self.device, self.props) = zhinst.utils.create_api_session(
            device_id, self.api_level,
            required_devtype='HDAWG')
        self.awg_module = self.daq.awgModule()
        self.awg_module.set('awgModule/device', self.device)
        self.awg_module.execute()
        node_tree = self.download_device_node_tree()
        self.create_parameters_from_node_tree(node_tree)
        self.warnings_as_errors: List[str] = []
        self._compiler_sleep_time = 0.01

    def snapshot_base(self, update: bool = True,
                      params_to_skip_update: Optional[Sequence[str]] = None
                      ) -> Dict:
        """ Override the base method to ignore 'feature_code' by default."""
        params_to_skip = ['features_code']
        if params_to_skip_update is not None:
            params_to_skip += list(params_to_skip_update)
        return super(ZIHDAWG8, self).snapshot_base(update=update,
                                                   params_to_skip_update=params_to_skip)

    def snapshot(self, update=True):
        """ Override base method to make update default True."""
        return super(ZIHDAWG8, self).snapshot(update)

    def enable_channel(self, channel_number: int) -> None:
        """
        Enable a signal output, turns on a blue LED on the device.

        Args:
            channel_number: Output channel that should be enabled.
        """
        self.set('sigouts_{}_on'.format(channel_number), 1)

    def disable_channel(self, channel_number: int) -> None:
        """
        Disable a signal output, turns off a blue LED on the device.

        Args:
            channel_number: Output channel that should be disabled.
        """
        self.set('sigouts_{}_on'.format(channel_number), 0)

    def start_awg(self, awg_number: int):
        """
        Activate an AWG

        Args:
            awg_number: The AWG that should be enabled.
        """
        self.set('awgs_{}_enable'.format(awg_number), 1)

    def stop_awg(self, awg_number: int) -> None:
        """
        Deactivate an AWG

        Args:
            awg_number: The AWG that should be disabled.
        """
        self.set('awgs_{}_enable'.format(awg_number), 0)

    def waveform_to_csv(self, wave_name: str, *waveforms: list) -> None:
        """
        Write waveforms to a CSV file in the modules data directory so that it
        can be referenced and used in a sequence program. If more than one
        waveform is provided they will be played simultaneously but on separate
        outputs.

        Args:
            wave_name: Name of the CSV file, is used by a sequence program.
            waveforms: One or more waveforms that are to be written to a
                CSV file. Note if there are more than one waveforms then they
                have to be of equal length, if not the longer ones will be
                truncated.
        """
        data_dir = self.awg_module.getString('awgModule/directory')
        wave_dir = os.path.join(data_dir, "awg", "waves")
        if not os.path.isdir(wave_dir):
            raise Exception(
                "AWG module wave directory {} does not exist or is not a "
                "directory".format(
                    wave_dir))
        csv_file = os.path.join(wave_dir, wave_name + '.csv')
        with open(csv_file, "w", newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerows(zip(*waveforms))

    @staticmethod
    def generate_csv_sequence_program(wave_info: List[
        Tuple[int, Union[str, None], Union[str, None]]]) -> str:
        """
        A method that generates a sequence program that plays waveforms from
        csv files.

        Args:
            wave_info: A list of tuples containing information about the waves
                that are to be played. Every tuple should have a channel number
                and wave, marker or both wave and marker.

        Returns:
            A sequence program that can be compiled and uploaded.

        """
        awg_program = textwrap.dedent("""
                HEADER
                DECLARATIONS
                while(true){
                    playWave(WAVES);
                }
                """)
        sequence_header = '// generated by {}\n'.format(__name__)
        awg_program = awg_program.replace('HEADER', sequence_header)
        declarations = ZIHDAWG8._generate_declarations(wave_info)
        awg_program = awg_program.replace('DECLARATIONS', declarations)
        play_wave_arguments = ZIHDAWG8._get_waveform_arguments(wave_info)
        awg_program = awg_program.replace('WAVES', play_wave_arguments)
        return awg_program

    @staticmethod
    def _generate_declarations(wave_info):
        declarations = ""
        for _, wave, marker in wave_info:
            if wave is not None and marker is not None:
                declarations += ('wave {0} = "{0}";\n'.format(wave))
                declarations += ('wave {0} = "{0}";\n'.format(marker))
                declarations += ('{0} = {0} + {1};\n'.format(wave, marker))
            elif wave is not None:
                declarations += ('wave {0} = "{0}";\n'.format(wave))
            elif marker is not None:
                declarations += ('wave {0} = "{0}";\n'.format(marker))
        return declarations

    @staticmethod
    def _get_waveform_arguments(wave_info):
        argument_string = ('{}, {}' * len(wave_info)).replace('}{', '}, {')
        play_wave_arguments = []
        for channel, wave, marker in wave_info:
            play_wave_arguments.append(channel)
            wave = wave if wave is not None else marker
            play_wave_arguments.append(wave)
        return argument_string.format(*play_wave_arguments)

    def upload_sequence_program(self, awg_number: int,
                                sequence_program: str) -> int:
        """
        Uploads a sequence program to the device equivalent to using the the
        sequencer tab in the device's gui.

        Args:
            awg_number: The AWG that the sequence program will be uploaded to.
            sequence_program: A sequence program that should be played on the
                device.

        Returns:
            0 is Compilation was successful with no warnings.
            2 if Compilation was successful but with warnings.

        Raises:
            CompilerError: If error occurs during compilation of the sequence
                program, or if a warning is elevated to an error.
        """
        self.awg_module.set('awgModule/index', awg_number)
        self.awg_module.set('awgModule/compiler/sourcestring', sequence_program)
        while len(self.awg_module.get('awgModule/compiler/sourcestring')
                  ['compiler']['sourcestring'][0]) > 0:
            time.sleep(self._compiler_sleep_time)

        if self.awg_module.getInt('awgModule/compiler/status') == 1:
            raise CompilerError(
                self.awg_module.getString('awgModule/compiler/statusstring'))
        elif self.awg_module.getInt('awgModule/compiler/status') == 2:
            self._handle_compiler_warnings(
                self.awg_module.getString('awgModule/compiler/statusstring'))
        while self.awg_module.getDouble('awgModule/progress') < 1.0:
            time.sleep(self._compiler_sleep_time)

        return self.awg_module.getInt('awgModule/compiler/status')

    def _handle_compiler_warnings(self, status_string: str) -> None:
        warnings = [warning for warning in status_string.split('\n') if
                    re.search(WARNING_ANY, warning) is not None]
        errors = []
        for warning in warnings:
            for warning_as_error in self.warnings_as_errors:
                if re.search(warning_as_error, warning) is not None:
                    errors.append(warning)
            if warning not in errors:
                self.log.warning(warning)
        if len(errors) > 0:
            raise CompilerError('Warning treated as an error.', *errors)

    def upload_waveform(self, awg_number: int, waveform: list,
                        index: int) -> None:
        """
        Upload a waveform to the device memory at a given index.

        Note:
            There needs to be a place holder on the device as this only replaces
            a data in the device memory but does not allocate new memory space.

        Args:
            awg_number: The AWG where waveform should be uploaded to.
            waveform: An array of floating point values from -1.0 to 1.0, or
                integers in the range (-32768...+32768)
            index: Index of the waveform that will be replaced. If there are
                more than 1 waveforms used then the index corresponds to the
                position of the waveform in the Waveforms sub-tab of the AWG tab
                in the GUI.
        """
        self.set('awgs_{}_waveform_index'.format(awg_number), index)
        self.daq.sync()
        self.set('awgs_{}_waveform_data'.format(awg_number), waveform)

    def set_channel_grouping(self, group: int) -> None:
        """
        Set the channel grouping mode of the device.

        Args:
            group: 0: groups of 2. 1: groups of 4. 2: groups of 8 i.e., one
                sequencer program controls 8 outputs.
        """
        self.set('system_awg_channelgrouping', group)

    def create_parameters_from_node_tree(self, parameters: dict) -> None:
        """
        Create QuCoDeS parameters from the device node tree.

        Args:
            parameters: A device node tree.
        """
        for parameter in parameters.values():
            getter = partial(self._getter, parameter['Node'],
                             parameter['Type']) if 'Read' in parameter[
                'Properties'] else None
            setter = partial(self._setter, parameter['Node'],
                             parameter['Type']) if 'Write' in parameter[
                'Properties'] else False
            options = validators.Enum(
                *[int(val) for val in parameter['Options'].keys()]) \
                if parameter['Type'] == 'Integer (enumerated)' else None
            parameter_name = self._generate_parameter_name(parameter['Node'])
            self.add_parameter(name=parameter_name,
                               set_cmd=setter,
                               get_cmd=getter,
                               vals=options,
                               docstring=parameter['Description'],
                               unit=parameter['Unit']
                               )

    @staticmethod
    def _generate_parameter_name(node):
        values = node.split('/')
        return '_'.join(values[2:]).lower()

    def download_device_node_tree(self, flags: int = 0) -> dict:
        """
        Args:
            flags:
                ziPython.ziListEnum.settingsonly (0x08): Returns only nodes
                which are marked as setting
                ziPython.ziListEnum.streamingonly (0x10): Returns only
                streaming nodes
                ziPython.ziListEnum.subscribedonly (0x20): Returns only
                subscribed nodes
                ziPython.ziListEnum.basechannel (0x40): Return only one instance
                of a node in case of multiple channels

                Or any combination of flags can be used.

        Returns:
            A dictionary of the device node tree.
        """
        node_tree = self.daq.listNodesJSON('/{}/'.format(self.device), flags)
        return json.loads(node_tree)

    def _setter(self, name: str, param_type: str, value: Any) -> None:
        if param_type == "Integer (64 bit)" or \
                param_type == 'Integer (enumerated)':
            self.daq.setInt(name, value)
        elif param_type == "Double":
            self.daq.setDouble(name, value)
        elif param_type == "String":
            self.daq.setString(name, value)
        elif param_type == "ZIVectorData":
            self.daq.vectorWrite(name, value)

    def _getter(self, name: str, param_type: str) -> Any:
        if param_type == "Integer (64 bit)" or \
                param_type == 'Integer (enumerated)':
            return self.daq.getInt(name)
        elif param_type == "Double":
            return self.daq.getDouble(name)
        elif param_type == "String":
            return self.daq.getString(name)
        elif param_type == "ZIVectorData":
            return self.daq.getAsEvent(name)
