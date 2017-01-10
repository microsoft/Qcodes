import re
import logging
logging.basicConfig(level="DEBUG")

import zmq
# TODO(giulioungaretti)
# What you don’t want to collect 	How to avoid collecting it
# Information about where calls were made from.
# Set logging._srcfile to None. This avoids calling sys._getframe(), which
# may help to speed up your code in environments like PyPy (which can’t
# speed up code that uses sys._getframe()), if and when PyPy
# supports Python 3.x.
# TODO(giulioungaretti) review formatters


DEBUGF = "%(levelname)s:%(name)s:%(lineno)d:%(asctime)s%(message)s"
DATEFMT = "%H:%M:%S"


class QPUBHandler(logging.Handler):
    """A basic logging handler that emits log messages through a PUB socket.

    Takes an interface to connect to.

        handler = PUBHandler('inproc://loc')

    Log messages handled by this handler are broadcast with ZMQ topics
    ``this.root_topic`` comes first, followed by the log level
    (DEBUG,INFO,etc.), followed by any additional subtopics specified in the
    message by: log.debug("subtopic.subsub::the real message")
    """
    root_topic=""
    socket = None

    # add more info to debug it's not cheap but more informative
    formatters = {
        logging.DEBUG: logging.Formatter(DEBUGF, datefmt=DATEFMT),
        logging.INFO: logging.Formatter("%(levelname)s:%(name)s:%(message)s\n", datefmt=DATEFMT),
        logging.WARN: logging.Formatter(
            "%(levelname)s:%(filename)s:%(name)s:%(lineno)d - %(message)s\n", datefmt=DATEFMT),
        logging.ERROR: logging.Formatter(
            "%(levelname)s:%(filename)s:%(name)s:%(lineno)d - %(message)s - %(exc_info)s\n", datefmt=DATEFMT),
        logging.CRITICAL: logging.Formatter(
        "%(levelname)s:%(filename)s:%(lineno)d - %(message)s\n",  datefmt=DATEFMT)}

    def __init__(self, interface_or_socket, context=None):
        logging.Handler.__init__(self)
        self.ctx = context or zmq.Context()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.connect(interface_or_socket)

    def format(self, record):
        """Format a record."""
        values = parse(self.formatters[record.levelno]._fmt)
        json_out = {}
        for key in values:
            json_out[key] = getattr(record, key)
        return json_out


    def emit(self, record):
        """Emit a record message
        Args:
            record (logging.record): record to shovel on the socket
        """
        self.socket.send_json(self.format(record))


def parse(string):
    """Parses format string looking for substitutions"""
    standard_formatters = re.compile(r'\((.+?)\)', re.IGNORECASE)
    return standard_formatters.findall(string)


def check_broker(frontend_addres="tcp://*:5559", backend_address="tcp://*:5560"):
    """
    Simple and dumb check to see if  a  XPUB/XSUB broker exists.

    Args:
        frontend_addres (str): Interface to which the frontend is bound
        backend_address (str): Interface to which the backend is bound

    Returns:
        bool: Broker server exists

    """
    context = zmq.Context()
    # Socket facing clients
    frontend = context.socket(zmq.XSUB)
    f = True
    b = True
    try:
        frontend.bind(frontend_addres)
        f = False
    except zmq.error.ZMQError:
        pass

    # Socket facing services
    backend = context.socket(zmq.XPUB)
    try:
        backend.bind(backend_address)
        b = False
    except zmq.error.ZMQError:
        pass

    frontend.close()
    backend.close()
    context.term()
    return f and b
