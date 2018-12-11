import pickle
from typing import Dict
from threading import Thread

import pika
from pika.adapters.blocking_connection import BlockingConnection

from qcodes.dataset.rmq_setup import (read_config_file,
                                      setup_exchange_and_queues_from_conf)
from qcodes.dataset.data_storage_interface import (DataWriterInterface, VALUES)


class Heart(Thread):
    """
    A separate thread to send heartbeats to RMQ
    """

    def __init__(self, conn: BlockingConnection):
        super().__init__()
        self.conn = conn
        self.keep_beating = True
        self.conf = read_config_file()
        self.sleep_time = self.conf['heartbeat']/2

    def stop(self):
        self.keep_beating = False

    def run(self):
        while self.keep_beating:
            self.conn.sleep(self.sleep_time)
            self.conn.process_data_events()


class BakedPotato:
    """
    A non-functional heart for mocking
    """

    def __init__(self, conn: BlockingConnection):
        pass

    def stop(self):
        pass

    def start(self):
        pass


class RabbitMQWriterInterface(DataWriterInterface):

    def __init__(self, guid: str, disable_heartbeat: bool = False):
        super().__init__(guid)

        conf = read_config_file()
        conn, chan = setup_exchange_and_queues_from_conf(conf)
        self.conn = conn
        self.channel = chan

        if not disable_heartbeat:
            self.heart = Heart(self.conn)
        else:
            self.heart = BakedPotato(self.conn)

        self.heart.start()

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

    def close(self):
        self.heart.stop()
        self.conn.close()
