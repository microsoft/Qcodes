import pickle
from typing import Dict

import pika

from .data_storage_interface import (
    DataStorageInterface, VALUES)


class RabbitMQStorageInterface(DataStorageInterface):

    def __init__(self, guid: str, ds):
        super().__init__(guid)
        self._connection = pika.BlockingConnection(
            pika.ConnectionParameters('localhost'))
        self._channel = self._connection.channel()
        self._channel.exchange_declare(exchange='mydata',
                                       exchange_type='fanout')
        # the configuration of channes should perhaps be read from
        # a config file
        self._channel.queue_declare(queue='tordensky',
                                    durable=True, exclusive=False)
        self._channel.queue_bind(exchange='mydata', queue='tordensky')
        self._channel.queue_declare(queue='localstorage',
                                    durable=True, exclusive=False)
        self._channel.queue_bind(exchange='mydata', queue='localstorage')

    def store_results(self, results: Dict[str, VALUES]):
        results_dump = pickle.dumps(results)
        self._channel.publish(exchange='mydata',
                              routing_key='',
                              body=results_dump,
                              properties=pika.BasicProperties(
                                  # todo this should include the chunk id
                                  headers={'guid': self.guid},
                                  delivery_mode=2))

