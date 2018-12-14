from contextlib import contextmanager
from typing import Dict, List
from time import sleep

import pytest
import pika
from pika.exceptions import ConnectionClosed

from qcodes.dataset.rmq_setup import read_config_file, write_config_file
from qcodes.dataset.rabbitmq_storage_interface import RabbitMQWriterInterface


# TODO: perhaps this is not the most elegant way to detect RMQ?
try:
    conf = read_config_file()
    params = pika.ConnectionParameters(conf['host'])
    conn = pika.BlockingConnection(params)
    conn.close()
    RMQ_RUNNING = True
except ConnectionClosed:
    RMQ_RUNNING = False


@contextmanager
def configs_set_to(confs: List[Dict]):
    """
    Temporarily change certain configurations of rmq_conf.json and then reset
    them.

    Args:
        confs: List of dicts that will be passed to conf.update(), where
          conf is a copy of the original configurations dictionary.
    """

    original_conf = read_config_file()
    temp_conf = original_conf.copy()
    for change in confs:
        temp_conf.update(change)
    write_config_file(temp_conf)

    try:
        yield
    finally:
        write_config_file(original_conf)


def test_no_heartbeat_times_out():

    if not RMQ_RUNNING:
        pytest.skip('RMQ not running. Please start RMQ to run this test.')

    heartbeat = 1

    with configs_set_to([{"heartbeat": heartbeat}]):

        rmq_writer = RabbitMQWriterInterface(guid="testing",
                                             path_to_db='',
                                             disable_heartbeat=True)
        rmq_writer.store_results({'x': [1]})
        # TODO: why does this test fail when the sleep is less than
        # 5 * heartbeat ?
        sleep(5 * heartbeat)

        with pytest.raises(ConnectionClosed):
            rmq_writer.store_results({'x': [2]})

    rmq_writer.close()


def test_with_heartbeat_does_not_time_out():

    if not RMQ_RUNNING:
        pytest.skip('RMQ not running. Please start RMQ to run this test.')

    heartbeat = 1

    with configs_set_to([{"heartbeat": heartbeat}]):

        rmq_writer = RabbitMQWriterInterface(guid="testing",
                                             path_to_db='',
                                             disable_heartbeat=False)
        rmq_writer.store_results({'x': [1]})

        sleep(5 * heartbeat)

        rmq_writer.store_results({'x': [2]})

    rmq_writer.close()
