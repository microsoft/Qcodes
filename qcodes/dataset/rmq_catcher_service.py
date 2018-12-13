# This is the "queue to local storage service" that consumes messages and
# writes them to disk

import argparse
from typing import List, Dict, Optional
import time
import datetime
from functools import partial
import pickle

from qcodes.dataset.rmq_queue_consumer import QueueConsumer, RMQConsumer
from qcodes.dataset.data_storage_interface import (DataReaderInterface,
                                                   DataWriterInterface)
from qcodes.dataset.sqlite_storage_interface import (SqliteReaderInterface,
                                                     SqliteWriterInterface)
from qcodes.dataset.database import get_DB_location
from qcodes.dataset.sqlite_base import connect


class Catcher:
    """
    The object that integrates consumption of messages from the queue and
    writing those messages (subject to some logic) to local storage
    """

    def __init__(self, consumer: type,
                 consumer_callback: str = 'catcher_default',
                 reader: type = SqliteReaderInterface,
                 writer: type = SqliteWriterInterface,
                 consumer_kwargs: Optional[Dict] = None):

        consumer_kwargs = consumer_kwargs or {}

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

        if writer != SqliteWriterInterface:
            raise NotImplementedError('Catcher currently only supports '
                                      'writing to SQLite.')

        callbacks = {'consumer_default': None,
                     'catcher_default': self.message_callback}
        callback = callbacks[consumer_callback]

        self.consumer = consumer(callback=callback,
                                 **consumer_kwargs)
        self.reader_factory = reader
        self.writer_factory = writer

        self.active_writers: Dict[str: SqliteWriterInterface] = {}

        self.number_of_received_messages = 0

    def message_callback(self, ch, method, properties, body):
        self.number_of_received_messages += 1

        header = properties.headers
        guid = header['guid']

        print(properties.headers)
        print(pickle.loads(body))

        if guid not in self.active_guids:

            # Look up to see if run exists
            # if not, check if package is data, discard if it is
            # if it is metadata, see if it has enough info to create a run
            # if not, discard it

            # we assume that reader and writer are SQLite interfaces
            reader_conn = connect(get_DB_location())
            reader = self.reader_factory(guid, conn=reader_conn)
            writer_conn = connect(get_DB_location())
            if not reader.run_exists():
                self.active_writers[guid] = self.writer_factory(
                                                guid, conn=writer_conn)
                self.active_writers[guid].create_run()

        if header['messagetype'] == 'data':
            results = pickle.loads(body)
            self.active_writers[guid].store_results(results)

        if header['messagetype'] == 'metadata':
            metadata = pickle.loads(body)
            self.active_writers[guid].store_metadata(metadata)

    @property
    def active_guids(self) -> List[str]:
        return list(self.active_writers.keys())


def current_time() -> str:
    ts_raw = time.time()
    time_fmt = '%Y-%m-%d %H:%M:%S'
    ts = datetime.datetime.utcfromtimestamp(ts_raw).strftime(time_fmt)
    return ts


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=('Run the RMQ -> local store '
                                                  'service'))
    parser.add_argument('--callback', type=str,
                        help="The argument to pass for consumer_callback")
    args = parser.parse_args()
    cb = args.callback

    cb = cb or "catcher_default"

    catcher = Catcher(consumer=RMQConsumer, consumer_callback=cb)

    print(f'Started RabbitMQ Local Storage Consumer at {current_time()}')
    print('Press Ctrl-C to stop')
    try:
        catcher.consumer.start_consuming()
    except KeyboardInterrupt:
        catcher.consumer.close()
    finally:
        print(f'Closing RabbitMQ Local Storage Consumer at {current_time()}')
