import argparse
import json
from time import sleep
import logging

import zmq

from qcodes.dataset.data_set import load_by_id

log = logging.getLogger('writer')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Write data to dataset')
    parser.add_argument('port', metavar='port', type=int,
                        help='port to subscribe to for data')
    parser.add_argument('run_id', metavar='run_id', type=int,
                        help='run_id of dataset to write to')

    args = parser.parse_args()
    port = args.port
    run_id = args.run_id

    log.info(f'Was called with port {port} and run_id {run_id}\n')

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://127.0.0.1:{port}")

    topicfilter = str(run_id).encode('utf-8')
    socket.setsockopt(zmq.SUBSCRIBE, topicfilter)

    # according to "the book", we should sleep here
    sleep(1)

    # now signal back that we are ready to receive published data
    req_socket = context.socket(zmq.REQ)
    req_socket.connect(f"tcp://127.0.0.1:{port+1}")

    # send synchronisation request
    log.debug(f'Sending sync ping')
    req_socket.send(b"")

    # wait for publisher to ping back that it is ready
    req_socket.recv()
    log.debug(f'Got ping back')

    ds = load_by_id(run_id)

    log.info(f'Connected to dataset: {ds}\n')

    time_to_stop = False

    while not time_to_stop:
        raw_mssg = socket.recv().decode('utf-8')
        # The raw message is {topic},{message}
        topiclength = len(str(run_id))
        mssg = raw_mssg[topiclength+1:]

        log.debug(f"{mssg}\n")

        if mssg == "KILL":

            time_to_stop = True
            ds.mark_complete()
            socket.close()
        else:
            try:
                data = json.loads(mssg)
                ds.add_result(data)
            except Exception as e:
                log.error(f"{e}")

