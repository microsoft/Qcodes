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
                                                   DataWriterInterface,
                                                   MetaData)
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
                             f'{consumer}, which is not a '
                             'subclass of QueueConsumer')

        if not issubclass(reader, DataReaderInterface):
            raise ValueError('Received an invalid local storage reader, '
                             f'{reader}, which is not a '
                             'subclass of DataReaderInterface')

        if not issubclass(writer, DataWriterInterface):
            raise ValueError('Received an invalid local storage writer, '
                             f'{writer}, which is not a '
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

        self.active_writers: Dict[str, SqliteWriterInterface] = {}

        self.number_of_received_messages = 0

    def message_callback(self, ch, method, properties, body):
        self.number_of_received_messages += 1

        header = properties.headers
        guid = header['guid']
        message_type = header['messagetype']

        print(properties.headers)
        delivery_tag = method.delivery_tag

        if guid not in self.active_guids:

            # Look up to see if run exists
            # if not, check if package is data, discard if it is
            # if it is metadata, see if it has enough info to create a run
            # if not, discard it

            # we assume for now that reader and writer are SQLite interfaces
            reader_conn = connect(get_DB_location())
            reader = self.reader_factory(guid, conn=reader_conn)
            if not reader.run_exists():
                # Then the message must be metadata for us to continue
                if message_type != 'metadata':
                    # then we should put the message back in the queue
                    ch.basic_nack(delivery_tag=delivery_tag,
                                  requeue=True)
                else:
                    # parse the message
                    metadata = self.parse_metadata_body(body)
                    writer_conn = connect(header['db_location'])
                    self.active_writers[guid] = self.writer_factory(
                                                    guid, conn=writer_conn)

                    exp_id = metadata.tags['exp_id']
                    name = metadata.name
                    exp_name = metadata.exp_name
                    sample_name = metadata.sample_name
                    print(metadata)
                    self.active_writers[guid].create_run(
                        exp_id=exp_id, name=name, exp_name=exp_name,
                        sample_name=sample_name)
                    ch.basic_ack(delivery_tag=delivery_tag)


    def parse_metadata_body(self, body) -> MetaData:
        asdict = pickle.loads(body)
        return MetaData(**asdict)

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
    parser.add_argument('--testing', type=bool,
                        help='If True, consume from a non-durable test queue')
    parser.add_argument
    args = parser.parse_args()
    cb = args.callback
    testing = args.testing

    cb = cb or "catcher_default"

    catcher = Catcher(consumer=RMQConsumer,
                      consumer_callback=cb,
                      consumer_kwargs={'use_test_queue': testing})

    print(f'Started RabbitMQ Local Storage Consumer at {current_time()}')
    print('Press Ctrl-C to stop')
    try:
        catcher.consumer.start_consuming()
    except KeyboardInterrupt:
        catcher.consumer.close()
    finally:
        print(f'Closing RabbitMQ Local Storage Consumer at {current_time()}')
