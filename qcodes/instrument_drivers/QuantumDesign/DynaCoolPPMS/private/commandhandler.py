from typing import Any, Tuple, List
import logging
from collections import namedtuple

log = logging.getLogger(__name__)

try:
    import win32com.client
    import pythoncom
    from pythoncom import VT_BYREF, VT_R8, VT_I4
except ImportError as e:
    message = ('To use the DynaCool Driver, please install win32com.'
               ' Installation can be done with pip install pypiwin32com')
    log.exception(message)
    raise ImportError(message)


# Variable types
_variants = {'VT_R8': win32com.client.VARIANT(VT_BYREF | VT_R8, 0.0),
             'VT_I4': win32com.client.VARIANT(VT_BYREF | VT_I4, 0)}
CmdArgs = namedtuple('cmd_and_args', 'cmd args')


class CommandHandler:
    """
    This is the class that gets called by the server.py

    This class is responsible for making the actual calls into the instrument
    firmware. The idea is that this class get a SCPI-like string from the
    server, e.g. 'TEMP?' or 'TEMP 300, 10, 1' and then makes the corresponding
    MultiVu API call (or returns an error message to the server).
    """

    def __init__(self, inst_type: str='dynacool') -> None:
        pythoncom.CoInitialize()
        client_id = f'QD.MULTIVU.{inst_type.upper()}.1'
        try:
            self._mvu = win32com.client.Dispatch(client_id)
        except pythoncom.com_error:
            error_mssg = ('Could not connect to Multivu Application. Please '
                          'make sure that the Multi Application is running.')
            log.exception(error_mssg)
            raise ValueError(error_mssg)

        # Hard-code what we know about the MultiVu API
        self._gets = {'TEMP': CmdArgs(cmd=self._mvu.GetTemperature,
                                      args=[_variants['VT_R8'],
                                            _variants['VT_I4']]),
                      'CHAT': CmdArgs(cmd=self._mvu.GetChamberTemp,
                                      args=[_variants['VT_R8'],
                                            _variants['VT_I4']])}

        self._sets = {'TEMP': self._mvu.SetTemperature}

    def preparser(self, cmd_str: str) -> Tuple[CmdArgs, bool]:
        """
        Parse the raw SCPI-like input string into a list of strings

        Args:
            cmd_str: A SCPI-like string, e.g. 'TEMP?' or 'TEMP 300, 0.1, 1'

        Returns:
            A tuple of a CmdArgs tuple and a bool indicating whether this was
            a query
        """
        def err_func():
            return 'ERROR: unknown command'

        cmd_head = cmd_str[:4]

        if cmd_head not in set(self._gets.keys()).union(set(self._sets.keys())):
            cmd = err_func
            args: List[Any] = []
            is_query = False

        elif cmd_str.endswith('?'):
            cmd = self._gets[cmd_head].cmd
            args = self._gets[cmd_head].args
            is_query = True
        else:
            cmd = self._sets[cmd_head]
            args = list(float(arg) for arg in cmd_str[5:].split(', '))
            is_query = False

        output = (CmdArgs(cmd=cmd, args=args), is_query)

        return output

    def postparser(self, error_code: int, vals: List) -> str:
        """
        Parse the output of the MultiVu API call into a string that the server
        can send back to the client

        Args:
            inp: A tuple of (error_code, vals), where vals is a List of the
              returned values (empty in case of a set command)
        """
        response = f'{error_code}'

        for val in vals:
            response += f', {val}'

        return response

    def __call__(self, cmd: str) -> str:

        cmd_and_args, is_query = self.preparser(cmd)
        log.debug(f'Parsed {cmd} into {cmd_and_args}')

        # Actually perform the call into the MultiVu API
        error_code = cmd_and_args.cmd(*cmd_and_args.args)

        # return values in case we did a query
        if is_query:
            # read out the mutated values
            # (win32 reverses the order)
            vals = list(arg.value for arg in cmd_and_args.args)
            vals.reverse()
            # reset the value variables for good measures
            for arg in cmd_and_args.args:
                arg.value = 0
        else:
            vals = []

        response_message = self.postparser(error_code, vals)

        return response_message
