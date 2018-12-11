import pickle
from typing import Dict

import pika

from qcodes.dataset.rmq_setup import (read_config_file,
                                      setup_exchange_and_queues_from_conf)
from qcodes.dataset.data_storage_interface import (DataWriterInterface, VALUES)


class RabbitMQStorageInterface(DataWriterInterface):

    def __init__(self, guid: str):
        super().__init__(guid)

        conf = read_config_file()
        conn, chan = setup_exchange_and_queues_from_conf(conf)
        self.conn = conn
        self.channel = chan

    def store_results(self, results: Dict[str, VALUES]):
        results_dump = pickle.dumps(results)
        self.channel.publish(exchange='mydata',
                             routing_key='',
                             body=results_dump,
                             properties=pika.BasicProperties(
                                 # todo this should include the chunk id
                                 headers={'guid': self.guid},
                                 delivery_mode=2))

    # "junk" implementation of abstract methods

    def create_run(self):
        raise NotImplementedError

    def prepare_for_storing_results(self):
        raise NotImplementedError

    def store_meta_data(self):
        raise NotImplementedError

