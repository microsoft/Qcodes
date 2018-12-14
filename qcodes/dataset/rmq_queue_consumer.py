
from typing import Callable, Optional
from abc import ABC, abstractmethod

import pika
from pika.exceptions import IncompatibleProtocolError, ChannelClosed

from qcodes.dataset.rmq_setup import (read_config_file,
                                      setup_exchange_and_queues_from_conf)


class QueueConsumer(ABC):
    """
    Abstract class defining what a consumer (either RMQ or Mock) must do
    """

    def __init__(self, callback: Callable):
        self.callback = callback

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def start_consuming(self):
        pass


class RMQConsumer(QueueConsumer):
    """
    The object whose role it is to read messages off of the 'Local Storage'
    queue and pass them on via a callback function
    """
    def __init__(self, callback: Optional[Callable] = None,
                 use_test_queue: bool = False):



        conf = read_config_file(testing=use_test_queue)
        conn, channel = setup_exchange_and_queues_from_conf(conf)
        self.connection = conn
        self.channel = channel

        callback = callback or self.default_callback
        super().__init__(callback)

        self.channel.basic_qos(prefetch_count=1,
                                all_channels=True)
        self.channel.basic_consume(self.callback,
                                    queue='localstorage',
                                    no_ack=False)

    @staticmethod
    def default_callback(ch: pika.channel.Channel,
                         method: pika.spec.Basic.Deliver,
                         properties: pika.spec.BasicProperties,
                         body):
        print('Something happened')
        print(ch, '\n', method, '\n', properties, '\n', body)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def close(self):
        self.connection.close()

    def start_consuming(self):
        self.channel.start_consuming()

