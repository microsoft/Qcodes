from unittest import TestCase
import time
import multiprocessing as mp

from qcodes.instrument.server import InstrumentServer
from qcodes.instrument.base import Instrument
from qcodes.process.server import (QUERY_WRITE, QUERY_ASK, RESPONSE_OK,
                                   RESPONSE_ERROR)
from qcodes.utils.helpers import LogCapture


def schedule(queries, query_queue):
    """
        queries is a sequence of (delay, args)
    query_queue is a queue to push these queries to, with each one waiting
        its delay after sending the previous one
    """
    for delay, args in queries:
        time.sleep(delay)
        query_queue.put(args)


def run_schedule(queries, query_queue):
    p = mp.Process(target=schedule, args=(queries, query_queue))
    p.start()
    return p


def get_results(response_queue):
    time.sleep(0.05)  # wait for any lingering messages to the queues
    responses = []
    while not response_queue.empty():
        responses.append(response_queue.get())

    return responses


class Holder:
    shared_kwargs = ['where']
    name = 'J Edgar'
    parameters = {}
    functions = {}

    def __init__(self, server_name=None, **kwargs):
        self.kwargs = kwargs
        self.d = {}

    def get(self, key):
        return self.d[key]

    def set(self, key, val):
        self.d[key] = val

    def get_extras(self):
        return self.kwargs

    def _get_method_attrs(self):
        return {}

    def close(self):
        pass

    def connection_attrs(self, new_id):
        return Instrument.connection_attrs(self, new_id)


class TimedInstrumentServer(InstrumentServer):
    timeout = 2


class TestInstrumentServer(TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.query_queue = mp.Queue()
        cls.response_queue = mp.Queue()
        cls.error_queue = mp.Queue()

    @classmethod
    def tearDownClass(cls):
        del cls.query_queue
        del cls.response_queue
        del cls.error_queue

    def test_normal(self):
        # we really only need to test local here - as a server it's already
        # used in other tests, but only implicitly (and not covered as it's
        # in a subprocess)
        queries = (
            # add an "instrument" to the server
            (0.5, (QUERY_ASK, 'new_id',)),
            (0.01, (QUERY_ASK, 'new', (Holder, 0))),

            # some sets and gets that work
            (0.01, (QUERY_WRITE, 'cmd',
                    (0, 'set', 'happiness', 'a warm gun'), {})),
            (0.01, (QUERY_WRITE, 'cmd',
                    (0, 'set'), {'val': 42, 'key': 'the answer'})),
            (0.01, (QUERY_ASK, 'cmd', (0, 'get'), {'key': 'happiness'})),
            (0.01, (QUERY_ASK, 'cmd', (0, 'get', 'the answer',), {})),

            # then some that make errors
            # KeyError
            (0.01, (QUERY_ASK, 'cmd', (0, 'get', 'Carmen Sandiego',), {})),
            # TypeError (too many args) shows up in logs
            (0.01, (QUERY_WRITE, 'cmd', (0, 'set', 1, 2, 3), {})),
            # TypeError (unexpected kwarg) shows up in logs
            (0.01, (QUERY_WRITE, 'cmd', (0, 'set', 'do'), {'c': 'middle'})),

            # and another good one, just so we know it still works
            (0.01, (QUERY_ASK, 'cmd', (0, 'get_extras'), {})),

            # delete the instrument and stop the server
            # (no need to explicitly halt)
            (0.01, (QUERY_ASK, 'delete', (0,)))
        )
        extras = {'where': 'infinity and beyond'}

        run_schedule(queries, self.query_queue)

        try:
            with LogCapture() as logs:
                TimedInstrumentServer(self.query_queue, self.response_queue,
                                      extras)
        except TypeError:
            from traceback import format_exc
            print(format_exc())

        self.assertEqual(logs.value.count('TypeError'), 2)
        for item in ('1, 2, 3', 'middle'):
            self.assertIn(item, logs.value)

        responses = get_results(self.response_queue)

        expected_responses = [
            (RESPONSE_OK, 0),
            (RESPONSE_OK, {
                'functions': {},
                'id': 0,
                'name': 'J Edgar',
                '_methods': {},
                'parameters': {}
            }),
            (RESPONSE_OK, 'a warm gun'),
            (RESPONSE_OK, 42),
            (RESPONSE_ERROR, ('KeyError', 'Carmen Sandiego')),
            (RESPONSE_OK, extras)
        ]
        for response, expected in zip(responses, expected_responses):
            if expected[0] == RESPONSE_OK:
                self.assertEqual(response, expected)
            else:
                self.assertEqual(response[0], expected[0])
                for item in expected[1]:
                    self.assertIn(item, response[1])
