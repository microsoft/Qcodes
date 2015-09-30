import asyncio
import time
from datetime import datetime

from qcodes.instrument.base import BaseInstrument


class MockInstrument(BaseInstrument):
    def __init__(self, name, delay=0, model=None, keep_history=True,
                 use_async=False, read_response=None, **kwargs):
        '''
        Creates a software instrument, for modeling or testing
        inputs:
            name: (string) the name of this instrument
            delay: the time (in seconds) to wait after any operation
                to simulate communication delay
            model: an object with write and ask methods, taking 2 or 3 args:
                instrument: the name of the instrument
                parameter: the name of the parameter
                value (write only): the value to write, as a string

        parameters to pass to model should be declared with:
            get_cmd = param_name + '?'
            set_cmd = param_name + ' {:.3f}' (specify the format & precision)
        alternatively independent functions may still be provided.
        '''
        super().__init__(name, **kwargs)

        if not isinstance(delay, (int, float)) or delay < 0:
            raise TypeError('delay must be a non-negative number')
        self._delay = delay

        # try to access write and ask so we know they exist
        model.write
        model.ask
        self._model = model

        # keep a record of every command sent to this instrument
        # for debugging purposes
        if keep_history:
            self.keep_history = True
            self.history = []

        # do we want a sync or async model?
        if use_async:
            self.write_async = self._write_async
            self.read_async = self._read_async
            self.ask_async = self._ask_async
        else:
            self.write = self._write
            self.read = self._read
            self.ask = self._ask

        # just for test purposes
        self._read_response = read_response

    def _write_inner(self, cmd):
        try:
            parameter, value = cmd.split(' ', 1)
        except ValueError:
            parameter, value = cmd, None  # for functions with no value

        if self.keep_history:
            self.history.append((datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                 'write', parameter, value))

        self._model.write(instrument=self.name, parameter=parameter,
                          value=value)

    def _write(self, cmd):
        if self._delay:
            time.sleep(self._delay)

        self._write_inner(cmd)

    @asyncio.coroutine
    def _write_async(self, cmd):
        if self._delay:
            yield from asyncio.sleep(self._delay)

        self._write_inner(cmd)

    def _ask_inner(self, cmd):
        parameter, blank = cmd.split('?')
        if blank:
            raise ValueError('text found after end of query')

        if self.keep_history:
            self.history.append((datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                 'ask', parameter))

        return self._model.ask(instrument=self.name, parameter=parameter)

    def _read(self):
        if self._delay:
            time.sleep(self._delay)

        return self._read_response

    @asyncio.coroutine
    def _read_async(self):
        if self._delay:
            yield from asyncio.sleep(self._delay)

        return self._read_response

    def _ask(self, cmd):
        if self._delay:
            time.sleep(self._delay)

        return self._ask_inner(cmd)

    @asyncio.coroutine
    def _ask_async(self, cmd):
        if self._delay:
            yield from asyncio.sleep(self._delay)

        return self._ask_inner(cmd)
