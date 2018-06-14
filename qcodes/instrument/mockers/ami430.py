import re
import time
from datetime import datetime


class MockAMI430:
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

    ramp_rate_units = {
        "A/s": "0",
        "A/min": "1"
    }

    quench_state = {False: "0", True: "1"}

    def __init__(self, name):

        self.name = name
        self.log_messages = []

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

    @staticmethod
    def message_parser(gs, msg_str, key):
        """
        * If gs = "get":
        Let suppose key = "RAMP:RATE:UNITS", then if we get msg_str =
        "RAMP:RATE:UNITS?" then match will be True and args = None. If
        msg_str = "RAMP:RATE:UNITS:10?" then match = True and args =
        "10". On the other hand if key = "RAMP" then both
        "RAMP:RATE:UNITS?" and "RAMP:RATE:UNITS:10?" will cause match
        to be False

        * If gs = "set"
        If key = "STATE" and msg_str = "STATE 2,1" then match = True
        and args = "2,1". If key="STATE" and msg_str = STATE:ELSE 2,1
        then match is False.

        Consult [1] for a complete description of the AMI430 protocol.

        [1]
        http://www.americanmagnetics.com/support/manuals/mn-4Q06125PS-430.pdf

        Args: gs (string): "get", or "set" msg_str (string): the
            message string the mock instrument gets.  key (string):
            one of the keys in self.handlers

        Returns: match (bool): if the key and the msg_str match, then
            match = True args (string): if any arguments are present
            in the message string these will be passed along. This is
            always None when match = False

        """
        # If the message string matches a key exactly we have a match
        # with no arguments
        if msg_str == key:
            return True, None

        # We use regular expressions to find out if the message string
        # and the key match. We need to replace reserved regular
        # expression characters in the key. For instance replace
        # "*IDN" with "\*IDN".
        reserved_re_characters = "\^${}[]().*+?|<>-&"
        for c in reserved_re_characters:
            key = key.replace(c, "\{}".format(c))

        # Get and set messages use different regular expression
        s = {"get": "(:[^:]*)?\?$", "set": "([^:]+)"}[gs]
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

    def _getter(self, attribute):
        return lambda _: getattr(self, attribute)

    def _setter(self, attribute):
        return lambda value: setattr(self, attribute, value)

    def _log(self, msg):

        now = datetime.now()
        log_line = "[{}] {}: {}".format(now.strftime("%d:%m:%Y-%H:%M:%S.%f"),
                                        self.name, msg)
        self.log_messages.append(log_line)

    def _handle_messages(self, msg):
        """
        Args:
            msg (string): a message received through the socket
                communication layer

        Returns:
            rval (string or None): If the type of message requests a
                value (a get message) then this value is returned by this
                function. A set message will return a None value.
        """

        # A "get" message ends with a "?" and will invoke the get
        # part of the handler defined in self.handlers.
        gs = {True: "get", False: "set"}[msg.endswith("?")]

        rval = None
        handler = None

        # Find which handler is suitable to handle the message
        for key in self.handlers:
            match, args = MockAMI430.message_parser(gs, msg, key)
            if not match:
                continue

            handler = self.handlers[key][gs]
            if callable(handler):
                rval = handler(args)
            else:
                rval = handler

            break

        if handler is None:
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

    def get_log_messages(self):
        return self.log_messages

    def ask(self, msg):
        return self._handle_messages(msg)

    def write(self, msg):
        self._handle_messages(msg)
