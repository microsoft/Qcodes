# This is the "queue to local storage service" that consumes messages and
# writes them to disk

import argparse
from typing import List, Dict, Optional
import time
import datetime

from qcodes.dataset.rmq_queue_consumer import QueueConsumer, RMQConsumer
from qcodes.dataset.data_storage_interface import (DataReaderInterface,
                                                   DataWriterInterface)
from qcodes.dataset.sqlite_storage_interface import (SqliteReaderInterface,
                                                     SqliteWriterInterface)


class Catcher:
    """
    The object that integrates consumption of messages from the queue and
    writing those messages (subject to some logic) to local storage
    """

    def __init__(self, consumer: type,
                 reader: type = SqliteReaderInterface,
                 writer: type = SqliteWriterInterface,
                 consumer_kwargs: Optional[Dict] = None,
                 reader_kwargs: Optional[Dict] = None,
                 writer_kwargs: Optional[Dict] = None):

        consumer_kwargs = consumer_kwargs or {}
        reader_kwargs = reader_kwargs or {}
        writer_kwargs = writer_kwargs or {}

        if not issubclass(consumer, QueueConsumer):
            raise ValueError('Received an invalid consumer, '
                             f'f{consumer}, which is not a '
                             'subclass of QueueConsumer')

        if not issubclass(reader, DataReaderInterface):
            raise ValueError('Received an invalid local storage reader, '
                             f'f{reader}, which is not a '
                             'subclass of DataReaderInterface')

        if not issubclass(writer, DataWriterInterface):
            raise ValueError('Received an invalid local storage writer, '
                             f'f{writer}, which is not a '
                             'subclass of DataWriterInterface')

        self.consumer = consumer(callback=self.message_callback,
                                 **consumer_kwargs)
        self.reader_factory = reader
        self.writer_factory = writer

        self.active_guids: List[str] = []

        self.number_of_received_messages = 0

    # TODO: the extension of this function is the real development.
    # For now allow a simple action that is testable (and that we'll probably
    # want to keep)

    def message_callback(self, ch, method, properties, body):
        self.number_of_received_messages += 1


def current_time() -> str:
    ts_raw = time.time()
    time_fmt = '%Y-%m-%d %H:%M:%S'
    ts = datetime.datetime.utcfromtimestamp(ts_raw).strftime(time_fmt)
    return ts


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=('Run the RMQ -> local store '
                                                  'service'))
    args = parser.parse_args()

    catcher = Catcher(consumer=RMQConsumer)

    print(f'Started RabbitMQ Local Storage Consumer at {current_time()}')
    print('Press Ctrl-C to stop')
    try:
        catcher.consumer.start_consuming()
    except KeyboardInterrupt:
        catcher.consumer.close()
    finally:
        print(f'Closing RabbitMQ Local Storage Consumer at {current_time()}')
