import time
import json
import zmq

_LINGER = 1000  # milliseconds
_ZMQ_HWM = int(5e8 / 120) # 500MB max memory for the logger

class UnboundedPublisher:
    """
    UnBounded publisher.
    Use with care as it will use as much memory as needed (meaning all of it).
    NOTE that this offers no guarantees on message delivery.
    If there is no reciever the message is LOST.
    """

    def __init__(self,
                 topic: str,
                 interface_or_socket: str="tcp://localhost:5559",
                 context: zmq.Context = None) -> None:
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


class Publisher(UnboundedPublisher):
    """
    Publisher.
    Allows for a publisher that will not use all the memory.
    Tune the timeout and hwm to fit the needs of the situation.
    We start with very permissive defaults:

        - 10 seconds linger
        - 2.5 GB cache

    NOTE that this offers no guarantees on message delivery.
    If there is no reciever the message is LOST.
    """

    def __init__(self, topic: str,
                 interface_or_socket: str="tcp://localhost:5559",
                 timeout: int = _LINGER*10,
                 hwm: int = _ZMQ_HWM*5,  context: zmq.Context = None) -> None:
        """

        Args:
            interface_or_socket:  Interface or socket to connect to
            topic: Topic of this publisher
            timeout: time in millisecond to wait before destroying this
                    published and the messages it caches
            hwm: number of messages to keep in the cache
            context: Context to reuse if desired
        """
        super().__init__(topic, interface_or_socket, context)
        self.socket.setsockopt(zmq.LINGER, timeout)
        self.socket.set_hwm(hwm)

    def send(self, msg: object):
        # Sleep for a nS to avoid going at max speed
        # the reason for this sleep is to "try" to not lose messages
        # realistically nothing will send small messages at an higher frequency
        # if it happens, then we may as well stop using python!
        time.sleep(1e-09)
        self.socket.send_multipart([self.topic, json.dumps(msg).encode()])
