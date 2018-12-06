# This is the "queue to local storage service" that consumes messages and
# writes them to disk

import argparse
from typing import List, Dict
import time
import datetime

from qcodes.dataset.rmq_queue_consumer import QueueConsumer, RMQConsumer
from qcodes.dataset.data_storage_interface import DataStorageInterface
from qcodes.dataset.sqlite_storage_interface import SqliteStorageInterface


class Catcher:
    """
    The object that integrates consumption of messages from the queue and
    writing those messages (subject to some logic) to local storage
    """

    def __init__(self, consumer: type,
                 local_reader_writer: type,
                 consumer_kwargs: Dict = {}):

        if not issubclass(consumer, QueueConsumer):
            raise ValueError('Received an invalid consumer, '
                             f'f{consumer}, which is not a '
                             'subclass of QueueConsumer')

        # NB: as soon as possible, we should split the DataStorageInterface
        # into a reader and a writer and use those here
        if not issubclass(local_reader_writer, DataStorageInterface):
            raise ValueError('Received an invalid local_reader_writer, '
                             f'f{local_reader_writer}, which is not a '
                             'subclass of DataStorageInterface')

        self.consumer = consumer(callback=self.message_callback,
                                 **consumer_kwargs)
        self.rw_factory = local_reader_writer

        self.active_guids: List[str] = []

        self.number_of_received_messages = 0

    # TODO: the extension of this function is the real development.
    # For now allow a simple action that is testable (and that we'll probably
    # want to keep)

    def message_callback(self, ch, method, properties, body):
        self.number_of_received_messages += 1


def current_time() -> str:
    ts_raw = time.time()
    ts = datetime.utcfromtimestamp(ts_raw).strftime('%Y-%m-%d %H:%M:%S')
    return ts


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=('Run the RMQ -> local store '
                                                  'service'))
    args = parser.parse_args()

    catcher = Catcher(consumer=RMQConsumer,
                      local_reader_writer=SqliteStorageInterface)

    print(f'Started RabbitMQ Local Storage Consumer at {current_time()}')
    print('Press Ctrl-C to stop')
    try:
        catcher.consumer.start_consuming()
    except KeyboardInterrupt:
        catcher.consumer.close()
    finally:
        print(f'Closing RabbitMQ Local Storage Consumer at {current_time()}')
