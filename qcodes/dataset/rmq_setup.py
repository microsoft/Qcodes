# This script sets up the RMQ with the exchanges and queues specified in the
# rmq_conf.json file (or rmq_test_conf.json if we are testing)
from typing import Dict, Tuple
import json
import argparse

import qcodes
import pika
from pika.exceptions import ConnectionClosed
from pika.adapters.blocking_connection import (BlockingConnection,
                                               BlockingChannel)


def validate_conf(conf: Dict) -> None:
    """
    Validate the configurations and give a friendly error message in case the
    configurations are invalid
    """
    raise NotImplementedError


def setup_exchange_and_queues_from_conf(
        conf: Dict) -> Tuple[BlockingConnection, BlockingChannel]:
    """
    Setup the exchange and the queues based on the information in the conf

    Args:
        conf: the json.loads of the conf file
    """
    params = pika.ConnectionParameters(conf['host'],
                                       heartbeat=conf['heartbeat'])
    try:
        # TODO: should we use a blockingconnection here?
        conn = pika.BlockingConnection(params)
    except ConnectionClosed as e:
        raise RuntimeError('No running RMQ found. Did you forget to run '
                           'docker_setup.py?') from e
    channel = conn.channel()

    for exchange in conf['exchanges']:
        channel.exchange_declare(**exchange)

    for queue in conf['queues']:
        channel.queue_declare(**queue)

    for binding in conf['bindings']:
        channel.queue_bind(**binding)

    return conn, channel


def read_config_file(testing: bool = False) -> Dict:
    """
    Read the config file and return it as a dict
    """
    if not testing:
        filename = 'rmq_conf.json'
    else:
        filename = 'rmq_test_conf.json'

    conf_path = qcodes.dataset.__file__.replace('__init__.py', filename)

    with open(conf_path, 'r') as fid:
        raw_json = fid.read()
    conf = json.loads(raw_json)

    # TODO: validate conf
    return conf


def write_config_file(conf: Dict) -> None:
    """
    Write the given dictionary to the config file
    """
    # TODO: validate conf
    conf_path = qcodes.dataset.__file__.replace('__init__.py', 'rmq_conf.json')
    with open(conf_path, 'w') as fid:
        raw_json = json.dumps(conf, indent=4)
        fid.write(raw_json)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prepare queues and exchange')
    parser.add_argument('--testing', type=bool,
                        help='If True, set up a non-durable test queue')
    args = parser.parse_args()

    conf = read_config_file(testing=args.testing)

    print('[*] Setting up exchange and queues...')

    conn, chan = setup_exchange_and_queues_from_conf(conf)
    conn.close()

    print('[+] Done.')
