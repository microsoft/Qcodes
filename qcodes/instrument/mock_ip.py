"""
A simple module for mocking IP instruments.
"""

import re
import select
import socket
import sys
import time
from datetime import datetime
from queue import Queue
from threading import Thread


class StdOutQueue(Queue):
    """
    This class will allow us to redirect stdout to a Queue, which can be handy to test if correct output is
    printed to screen. This is also handy for inter-thread communication.
    """

    def __init__(self, *args, **kwargs):
        Queue.__init__(self, *args, **kwargs)

    def write(self, msg):
        self.put(msg)

    def flush(self):
        sys.__stdout__.flush()


class MockIPInstrument(object):
    def __init__(self, name, port, ip_address="127.0.0.1", output_stream=None):
        self.name = name
        self.ip_address = ip_address
        self.port = port

        if output_stream in ["stdout", None]:
            self._output_stream = sys.stdout
        else:
            if hasattr(output_stream, "write"):
                self._output_stream = output_stream
                self._output_stream_is_file = False
            else:
                try:
                    self._output_stream = open(output_stream, "w")
                    self._output_stream_is_file = True
                except FileNotFoundError:
                    raise ValueError("output stream needs to either have a write method or be a file path")

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error:
            self.error = 'Failed to create socket'

        self.error = "Ok"
        self.__main__thread__ = None
        self.quit = False

    def start(self):
        self.socket.bind((self.ip_address, self.port))
        self.__main__thread__ = Thread(target=self._main_thread_)
        self.__main__thread__.start()
        return self

    def stop(self):
        self.quit = True
        self.__main__thread__.join()
        if self._output_stream_is_file:
            self._output_stream.close()

    def _getter(self, attribute):
        return lambda _: getattr(self, attribute)

    def _setter(self, attribute):
        return lambda value: setattr(self, attribute, value)

    def _log(self, msg):
        now = datetime.now()
        log_msg = "[{}] {}: {}".format(now.strftime("%d:%m:%Y-%H:%M:%S.%f"), self.name, msg)

        if self._output_stream is not None:
            self._output_stream.write(log_msg)

    def _main_thread_(self):

        self.socket.listen(1)
        socket_read_list = [self.socket]

        while not self.quit:
            readable, writable, errored = select.select(socket_read_list, [], [], 0.01)

            if not len(readable):
                continue

            for count, r in enumerate(readable):

                if r is self.socket:
                    c, addr = r.accept()
                    self._log("sending connected message")
                    c.send("Mock IP Device".encode())
                    socket_read_list += [c]
                    continue

                try:
                    msgs = r.recv(100).decode()

                    for msg in msgs.strip().split("\r\n"):
                        self._log("Received message: {}".format(msg))
                        response = self._handle_messages(msg)

                        if response is not None:
                            r.send("{}\r\n".format(response).encode())
                            self._log("Sending message: {}".format(response))

                except ConnectionResetError:
                    self._log("Got a connection reset error, removing connection from list")
                    socket_read_list.remove(r)

    def _handle_messages(self, msg):
        raise NotImplementedError("This needs to be implemented in sub-classes")


class MockAMI430(MockIPInstrument):
    states = {
        "RAMPING to target field/current": "1",
        "HOLDING at the target field/current": "2",
        "PAUSED": "3",
        "Ramping in MANUAL UP mode": "4",
        "Ramping in MANUAL DOWN mode": "5",
        "ZEROING CURRENT (in progress)": "6",
        "Quench detected": "7",
        "At ZERO current": "8",
        "Heating persistent switch": "9",
        "Cooling persistent switch": "10"
    }

    field_units = {
        "tesla": "1",
        "kilogauss": "0"
    }

    ramp_rate_units = {  # I made this up! Could not find correct values in manual
        "A/s": "0",
        "A/min": "1"
    }

    quench_state = {False: "0", True: "1"}

    def __init__(self, name, port, ip_address="127.0.0.1", output_stream="stdout"):

        self._field_mag = 0
        self._field_target = 0
        self._state = MockAMI430.states["HOLDING at the target field/current"]

        self.handlers = {
            "RAMP:RATE:UNITS": {
                "get": MockAMI430.ramp_rate_units["A/s"],
                "set": None
            },
            "FIELD:UNITS": {
                "get": MockAMI430.field_units["tesla"],
                "set": None
            },
            "*IDN": {
                "get": "v0.1 Mock",
                "set": None
            },
            "STATE": {
                "get": self._getter("_state"),
                "set": self._setter("_state")
            },
            "FIELD:MAG": {
                "get": self._getter("_field_mag"),
                "set": None
            },
            "QU": {
                "get": MockAMI430.quench_state[False],  # We are never in a quenching state so always return the
                # same value
                "set": None
            },
            "PERS": {
                "get": "0",
                "set": None
            },
            "PAUSE": {
                "get": self._is_paused,
                "set": self._do_pause
            },
            "CONF:FIELD:TARG": {
                "get": None,  # To get the field target, send a message "FIELD:TARG?"
                "set": self._setter("_field_target")
            },
            "FIELD:TARG": {
                "get": self._getter("_field_target"),
                "set": None
            },
            "PS": {
                "get": "0",  # The heater is off
                "set": None
            },
            "RAMP": {
                "set": self._do_ramp,
                "get": None
            },
            "RAMP:RATE:CURRENT": {
                "get": "0.1000,50.0000",
                "set": None
            },
            "COIL": {
                "get": "1",
                "set": None
            }
        }

        super(MockAMI430, self).__init__(name, port, ip_address=ip_address, output_stream=output_stream)

    @staticmethod
    def message_parser(gs, msg_str, key):
        """
        * If gs = "get":
        Let suppose key = "RAMP:RATE:UNITS", then if we get msg_str = "RAMP:RATE:UNITS?" then match will be True and
        args = None. If msg_str = "RAMP:RATE:UNITS:10?" then match = True and args = "10". On the other hand if
        key = "RAMP" then both "RAMP:RATE:UNITS?" and "RAMP:RATE:UNITS:10?" will cause match to be False

        * If gs = "set"
        If key = "STATE" and msg_str = "STATE 2,1" then match = True and args = "2,1". If key="STATE" and
        msg_str =  STATE:ELSE 2,1 then match is False.

        Consult [1] for a complete description of the AMI430 protocol.

        [1] http://www.americanmagnetics.com/support/manuals/mn-4Q06125PS-430.pdf

        :param gs: string, "get", or "set"
        :param msg_str: string, the message string the mock instrument gets through the network socket.
        :param key: string, one of the keys in self.handlers
        :return: (match, args), (bool, string), if the key and the msg_str match, then match = True. If any arguments
                                                are present in the message string these will be passed along.
        """
        if msg_str == key:  # If the message string matches a key exactly we have a match with no arguments
            return True, None

        # We use regular expressions to find out if the message string and the key match. We need to replace reserved
        # regular expression characters in the key. For instance replace "*IDN" with "\*IDN".
        reserved_re_characters = "\^${}[]().*+?|<>-&"
        for c in reserved_re_characters:
            key = key.replace(c, "\{}".format(c))

        s = {"get": "(:[^:]*)?\?$", "set": "([^:]+)"}[gs]  # Get and set messages use different regular expression
        # patterns to determine a match
        search_string = "^" + key + s
        r = re.search(search_string, msg_str)
        match = r is not None

        args = None
        if match:
            args = r.groups()[0]
            if args is not None:
                args = args.strip(":")

        return match, args

    def _handle_messages(self, msg):
        """
        :param msg: string, a message received through the socket communication layer
        :return: string or None,    If the type of message requests a value (a get message) then this value is returned
                                    by this function. A set message will return a None value.
        """

        gs = {True: "get", False: "set"}[msg.endswith("?")]  # A "get" message ends with a "?" and will invoke the get
        # part of the handler defined in self.handlers.

        rval = None
        for key in self.handlers:  # Find which handler is suitable to handle the message
            match, args = MockAMI430.message_parser(gs, msg, key)
            if not match:
                continue

            handler = self.handlers[key][gs]
            if callable(handler):
                rval = handler(args)
            else:
                rval = handler

            break

        if rval is None:
            self._log("Command {} unknown".format(msg))

        return rval

    def _do_pause(self, _):
        self._state = MockAMI430.states["PAUSED"]

    def _is_paused(self):
        return self._state == MockAMI430.states["PAUSED"]

    def _do_ramp(self, _):
        self._log("Ramping to {}".format(self._field_target))
        self._state = MockAMI430.states["RAMPING to target field/current"]
        time.sleep(0.1)  # Lets pretend to be ramping for a bit
        self._field_mag = self._field_target
        self._state = MockAMI430.states["HOLDING at the target field/current"]
