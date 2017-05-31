import re
import logging
import json

import zmq

logging.basicConfig(level="DEBUG")
DEBUGF = "%(levelname)s:%(name)s:%(lineno)d:%(asctime)s%(message)s"
INFOF = "%(levelname)s:%(name)s:%(message)s\n"
WARNF = "%(levelname)s:%(filename)s:%(name)s:%(lineno)d - %(message)s\n"
ERRORF = "%(levelname)s:%(filename)s:%(name)s:%(lineno)d - %(message)s - %(exc_info)s\n"
CRITICALF = "%(levelname)s:%(filename)s:%(name)s:%(lineno)d - %(message)s - %(exc_info)s\n"
DATEFMT = "%H:%M:%S"

_ALMOST_RANDOM_GUESS_LOG_MSG_SIZE = 120  # B per log message
_ZMQ_HWM = int( 5e8 / 120 ) # 500MB max memory for the logger
_LINGER = 1000  # milliseconds


class QPUBHandler(logging.Handler):
    """A basic logging handler that emits log messages through a PUB socket.

    Takes an interface to connect to.

        handler = PUBHandler('inproc://loc')

    Log messages handled by this handler are broadcast with ZMQ topic
    ``log.name`` (which is the name of the module, when logging is done right).

    This handler also has sane defaults when it comes to caching.

    By default it sets:
        - no more than 500MB (ish) message cache
        - messages have at best a second of lifetime

    This means:
        - if more than 500MB are cached, NEW messages are dropped
        - if nobody reads the messages after a second the socket will close
           wihtout waiting
    """
    # note that if we want zmq topcis the format MUST include name
    formatters = {
        logging.DEBUG: logging.Formatter(DEBUGF, datefmt=DATEFMT),
        logging.INFO: logging.Formatter(INFOF, datefmt=DATEFMT),
        logging.WARN: logging.Formatter(
            WARNF, datefmt=DATEFMT),
        logging.ERROR: logging.Formatter(
            ERRORF, datefmt=DATEFMT),
        logging.CRITICAL: logging.Formatter(
            CRITICALF, datefmt=DATEFMT),
    }

    def __init__(self, interface_or_socket, context=None):
        logging.Handler.__init__(self)
        self.ctx = context or zmq.Context()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.connect(interface_or_socket)
        # set-up sane defaults
        self.socket.setsockopt(zmq.LINGER, _LINGER)
        self.socket.set_hwm(_ZMQ_HWM)

    def format(self, record):
        """Format a record."""
        self.formatters[record.levelno].format(record)
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
        msg = self.format(record)
        _logger_name = msg.get("name", "")
        if _logger_name:
            topic = "logger" + "." + _logger_name
        else:
            topic = "logger"
        self.socket.send_multipart([topic.encode(), json.dumps(msg).encode()])


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


class Publisher():
    """
    Unthrottled publisher.
    Use with care as it will use as much memory as needed (meaning all of it)
    """

    def __init__(self, interface_or_socket: str,
                 topic: str, context: zmq.Context = None):
        """

        Args:
            interface_or_socket:  Interface or socket to connect to
            topic: Topic of this publisher
            context: Context to reuse if desired
        """
        self.ctx = context or zmq.Context()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.connect(interface_or_socket)
        self.topic = topic.encode()

    def send(self, msg: object):
        self.socket.send_multipart([self.topic, json.dumps(msg).encode()])


class ThottledPublisher(Publisher):
    """
    Trottled publisher.
    Allows for a publisher that will not use all the memory.
    Tune the timeout and hwm to fit the needs of the situation.
    """

    def __init__(self, interface_or_socket: str, topic: str,
                 timeout: int = _LINGER,
                 hwm: int = _ZMQ_HWM,  context: zmq.Context = None):
        """

        Args:
            interface_or_socket:  Interface or socket to connect to
            topic: Topic of this publisher
            timeout: time in millisecond to wait before destroying this published and the messages it caches
            hwm: number of messages to keep in the cache
            context: Context to reuse if desired
        """
        super().__init__(interface_or_socket, topic, context)
        self.socket.setsockopt(zmq.LINGER, timeout)
        self.socket.set_hwm(hwm)

    def send(self, msg: object):
        self.socket.send_multipart([self.topic, json.dumps(msg).encode()])
