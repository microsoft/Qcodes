"""
A simple module for mocking IP instruments. We will be referring a lot to:

[1] http://www.americanmagnetics.com/support/manuals/mn-4Q06125PS-430.pdf
"""

import select
import socket
import sys
import time
from datetime import datetime
from threading import Thread


class MockIPInstrument(object):
    def __init__(self, name, ip_address, port, log_file=None, max_connections=5, output_stream=None):
        self.name = name
        self.ip_address = ip_address
        self.port = port
        self.log_file = log_file
        self.max_connections = max_connections

        if output_stream == "stdout":
            self.output_stream = sys.stdout
        else:
            self.output_stream = output_stream

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

    def _log(self, msg):
        now = datetime.now()
        log_msg = "[{} {}: {}]".format(now, self.name, msg)

        if self.output_stream is not None:
            self.output_stream.write("{}: {}".format(self.name, msg))

        if self.log_file is not None:
            with open(self.log_file, "a") as fh:
                fh.write(log_msg + "\n")

    def _main_thread_(self):

        self.socket.listen(self.max_connections)
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
                        response = self._response(msg)

                        if response is not None:
                            r.send("{}\r\n".format(response).encode())
                            self._log("Sending message: {}".format(response))

                except ConnectionResetError:
                    self._log("Got a connection reset error, removing connection from list")
                    socket_read_list.remove(r)

    def _response(self, msg):
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

    def __init__(self, name, ip_address, port, log_file=None, max_connections=5, output_stream="stdout"):

        self.internal_state = {
            "RAMP:RATE:UNITS": MockAMI430.ramp_rate_units["A/s"],
            "FIELD:UNITS": MockAMI430.field_units["tesla"],
            "*IDN": "v0.1 Mock",
            "STATE": MockAMI430.states["HOLDING at the target field/current"],
            "FIELD:MAG": "0",  # We start out with zero field
            "QU": MockAMI430.quench_state[False],
            "PERS": "0",  # We do not start out in the persistent mode
            "PAUSE": self._do_pause,
            "CONF:FIELD:TARG": self._do_set_field_target,
            "FIELD:TARG": "0",  # The initial ramp target is zero
            "PS": "0",  # The heater is off in the beginning
            "RAMP": self._do_ramp
        }

        super(MockAMI430, self).__init__(name, ip_address, port, log_file=log_file, max_connections=max_connections,
                                         output_stream=output_stream)

    def _response(self, msg):

        if msg.endswith("?"):
            msg = msg.strip("?")
            return self._question_response(msg)
        else:
            self._action_response(msg)

    def _action_response(self, msg):

        parts = msg.split()

        if len(parts) == 2:
            f, arg = parts
        else:
            f, arg = msg, True

        if callable(self.internal_state[f]):
            if f == "RAMP":
                pass
            self.internal_state[f](arg)
        else:
            self.internal_state[f] = arg

    def _question_response(self, msg):

        if msg not in self.internal_state:
            self._log("Message {} not understood".format(msg))
        else:
            return self.internal_state[msg]

    def _do_pause(self, _):
        self.internal_state["STATE"] = MockAMI430.states["PAUSED"]

    def _do_set_field_target(self, value):
        # This function is triggered when a message CONF:FIELD:TARG is received.
        # We will set the ramp target when this happens.
        self.internal_state["FIELD:TARG"] = value

    def _do_ramp(self, _):
        self._log("Ramping to {}".format(self.internal_state["FIELD:TARG"]))

        self.internal_state["STATE"] = MockAMI430.states["RAMPING to target field/current"]
        # Lets pretend to be ramping for a bit
        time.sleep(0.1)
        self.internal_state["FIELD:MAG"] = self.internal_state["FIELD:TARG"]
        self.internal_state["STATE"] = MockAMI430.states["HOLDING at the target field/current"]