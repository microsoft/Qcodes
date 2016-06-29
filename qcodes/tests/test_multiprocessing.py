from unittest import TestCase, skipIf
import time
import re
import sys
import multiprocessing as mp
from queue import Empty
from unittest.mock import patch

import qcodes
from qcodes.process.helpers import set_mp_method, kill_queue
from qcodes.process.qcodes_process import QcodesProcess
from qcodes.process.stream_queue import get_stream_queue, _SQWriter
from qcodes.process.server import ServerManager, RESPONSE_OK, RESPONSE_ERROR
import qcodes.process.helpers as qcmp
from qcodes.utils.helpers import in_notebook, LogCapture
from qcodes.utils.timing import calibrate

BREAK_SIGNAL = '~~BREAK~~'


class sqtest_echo:
    def __init__(self, name, delay=0.01, has_q=True):
        self.q_out = mp.Queue()
        self.q_err = mp.Queue()
        self.p = QcodesProcess(target=sqtest_echo_f,
                               args=(name, delay, self.q_out, self.q_err,
                                     has_q),
                               name=name)
        self.p.start()
        self.delay = delay
        self.resp_delay = delay * 2 + 0.03

    def send_out(self, msg):
        self.q_out.put(msg)
        time.sleep(self.resp_delay)

    def send_err(self, msg):
        self.q_err.put(msg)
        time.sleep(self.resp_delay)

    def halt(self):
        if not (hasattr(self, 'p') and self.p.is_alive()):
            return
        self.q_out.put(BREAK_SIGNAL)
        self.p.join()
        time.sleep(self.resp_delay)
        for q in ['q_out', 'q_err']:
            if hasattr(self, q):
                queue = getattr(self, q)
                kill_queue(queue)
                kill_queue(queue)  # repeat just to make sure it doesn't error

    def __del__(self):
        self.halt()


def sqtest_echo_f(name, delay, q_out, q_err, has_q):
    while True:
        time.sleep(delay)

        if not q_out.empty():
            out = q_out.get()

            if out == BREAK_SIGNAL:
                # now test that disconnect works, and reverts to
                # regular stdout and stderr
                if has_q:
                    try:
                        get_stream_queue().disconnect()
                    except RuntimeError:
                        pass
                    print('stdout ', end='', flush=True)
                    print('stderr ', file=sys.stderr, end='', flush=True)
                break

            print(out, end='', flush=True)

        if not q_err.empty():
            print(q_err.get(), file=sys.stderr, end='', flush=True)


def sqtest_exception():
    raise RuntimeError('Boo!')


class TestMpMethod(TestCase):
    def test_set_mp_method(self):
        start_method = mp.get_start_method()
        self.assertIn(start_method, ('fork', 'spawn', 'forkserver'))

        # multiprocessing's set_start_method is NOT idempotent
        with self.assertRaises(RuntimeError):
            mp.set_start_method(start_method)

        # but ours is
        set_mp_method(start_method)

        # it will still error on gibberish, but different errors depending
        # on whether you force or not
        with self.assertRaises(RuntimeError):
            set_mp_method('spoon')
        with self.assertRaises(ValueError):
            set_mp_method('spoon', force=True)

        # change the error we look for to test strange error handling
        mp_err_normal = qcmp.MP_ERR
        qcmp.MP_ERR = 'who cares?'
        with self.assertRaises(RuntimeError):
            set_mp_method('start_method')
        qcmp.MP_ERR = mp_err_normal


class TestQcodesProcess(TestCase):
    def setUp(self):
        mp_stats = calibrate(quiet=True)
        self.MP_START_DELAY = mp_stats['mp_start_delay']
        self.MP_FINISH_DELAY = mp_stats['mp_finish_delay']
        self.SLEEP_DELAY = mp_stats['sleep_delay']
        self.BLOCKING_TIME = mp_stats['blocking_time']
        self.sq = get_stream_queue()

    @skipIf(getattr(qcodes, '_IN_NOTEBOOK', False),
            'called from notebook')
    def test_not_in_notebook(self):
        # below we'll patch this to True, but make sure that it's False
        # in the normal test runner.
        self.assertEqual(in_notebook(), False)

        # and make sure that processes run this way do not use the queue
        with self.sq.lock:
            p = sqtest_echo('hidden', has_q=False)
            time.sleep(self.MP_START_DELAY)
            p.send_out('should go to stdout;')
            p.send_err('should go to stderr;')
            p.halt()

            self.assertEqual(self.sq.get(), '')

    @patch('qcodes.process.qcodes_process.in_notebook')
    def test_qcodes_process_exception(self, in_nb_patch):
        in_nb_patch.return_value = True

        with self.sq.lock:
            name = 'Hamlet'
            p = QcodesProcess(target=sqtest_exception, name=name)

            initial_outs = (sys.stdout, sys.stderr)

            # normally you call p.start(), but for this test we want
            # the function to actually run in the main process
            # it will run the actual target, but will print the exception
            # (to the queue) rather than raising it.
            p.run()

            # output streams are back to how they started
            self.assertEqual((sys.stdout, sys.stderr), initial_outs)
            time.sleep(0.01)
            exc_text = self.sq.get()
            # but we have the exception in the queue
            self.maxDiff = None
            self.assertGreaterEqual(exc_text.count(name + ' ERR'), 5)
            self.assertEqual(exc_text.count('Traceback'), 1, exc_text)
            self.assertEqual(exc_text.count('RuntimeError'), 2)
            self.assertEqual(exc_text.count('Boo!'), 2)

    @patch('qcodes.process.qcodes_process.in_notebook')
    def test_qcodes_process(self, in_nb_patch):
        in_nb_patch.return_value = True

        queue_format = re.compile(
            '^\[\d\d:\d\d:\d\d\.\d\d\d p\d( ERR)?\] [^\[\]]*$')

        with self.sq.lock:
            p1 = sqtest_echo('p1')
            p2 = sqtest_echo('p2')
            time.sleep(self.MP_START_DELAY + p1.delay + p2.delay)

            self.assertEqual(self.sq.get(), '')

            procNames = ['<{}>'.format(name)
                         for name in ('p1', 'p2')]

            reprs = [repr(p) for p in mp.active_children()]
            for name in procNames:
                self.assertIn(name, reprs)

            # test each individual stream to send several messages on same
            # and different lines

            for sender, label, term in ([[p1.send_out, 'p1] ', ''],
                                         [p1.send_err, 'p1 ERR] ', '\n'],
                                         [p2.send_out, 'p2] ', '\n'],
                                         [p2.send_err, 'p2 ERR] ', '']]):
                sender('row row ')
                sender('row your boat\n')
                sender('gently down ')
                time.sleep(0.01)
                data = [line for line in self.sq.get().split('\n') if line]
                expected = [
                    label + 'row row row your boat',
                    label + 'gently down '
                ]
                # TODO - intermittent error here
                self.assertEqual(len(data), len(expected), data)
                for line, expected_line in zip(data, expected):
                    self.assertIsNotNone(queue_format.match(line), data)
                    self.assertEqual(line[14:], expected_line, data)

                sender(' the stream' + term)
                # no label/header as we're continuing  the previous line
                self.assertEqual(self.sq.get(), ' the stream' + term)

            p1.send_out('marco')
            p2.send_out('polo\n')  # we don't see these single terminators
            p1.send_out('marco\n')  # when we change streams
            p2.send_out('polo')
            time.sleep(0.01)

            data = self.sq.get().split('\n')
            for line in data:
                if line:
                    self.assertIsNotNone(queue_format.match(line))

            data_msgs = [line[14:] for line in data]
            expected = [
                '',
                'p1] marco',
                'p2] polo',
                'p1] marco',
                'p2] polo'
            ]
            self.assertEqual(data_msgs, expected)

            # Some OS's start more processes just for fun... so don't test
            # that p1 and p2 are the only ones.
            # self.assertEqual(len(reprs), 2, reprs)

            p1.halt()
            p2.halt()
            # both p1 and p2 should have finished now, and ended.
            reprs = [repr(p) for p in mp.active_children()]
            for name in procNames:
                self.assertNotIn(name, reprs)


class TestStreamQueue(TestCase):
    def test_connection(self):
        sq = get_stream_queue()
        sq.connect('')
        # TODO: do we really want double-connect to raise? or maybe
        # only raise if the process name changes?
        with self.assertRaises(RuntimeError):
            sq.connect('')
        sq.disconnect()
        with self.assertRaises(RuntimeError):
            sq.disconnect()

    def test_del(self):
        sq = get_stream_queue()
        self.assertTrue(hasattr(sq, 'queue'))
        self.assertTrue(hasattr(sq, 'lock'))
        self.assertIsNotNone(sq.instance)

        sq.__del__()

        self.assertFalse(hasattr(sq, 'queue'))
        self.assertFalse(hasattr(sq, 'lock'))
        self.assertIsNone(sq.instance)

        sq.__del__()  # just to make sure it doesn't error

    # this is basically tested in TestQcodesProcess, but the test happens
    # in a subprocess so coverage doesn't know about it. Anyway, there are
    # a few edge cases left that we have to test locally.
    def test_sq_writer(self):
        sq = get_stream_queue()
        with sq.lock:
            sq_clearer = _SQWriter(sq, 'Someone else')
            sq_clearer.write('Boo!\n')

            # apparently we need a little delay to make sure the queue
            # properly appears populated.
            time.sleep(0.01)

            sq.get()
            sq_name = 'A Queue'
            sqw = _SQWriter(sq, sq_name)

            # flush should exist, but does nothing
            sqw.flush()

            lines = [
                'Knock knock.\nWho\'s there?\n',
                'Interrupting cow.\n',
                'Interr-',
                'MOO!\n',
                ''
            ]

            for line in lines:
                sqw.write(line)

            time.sleep(0.01)

            # _SQWriter doesn't do any transformations to the messages, beyond
            # adding a time string and the stream name
            for line in lines:
                if not line:
                    self.assertEqual(sq.queue.empty(), True)
                    continue

                self.assertEqual(sq.queue.empty(), False)
                timestr, stream_name, msg = sq.queue.get()
                self.assertEqual(msg, line)
                self.assertEqual(stream_name, sq_name)

            # now test that even if the queue is unwatched the messages still
            # go there. If we're feeling adventurous maybe we can test if
            # something was actually printed.
            sqw.MIN_READ_TIME = -1
            new_message = 'should get printed\n'
            sqw.write(new_message)

            time.sleep(0.01)

            self.assertEqual(sq.queue.empty(), False)
            self.assertEqual(sq.queue.get()[2], new_message)

            # test that an error in writing resets stdout and stderr
            # nose uses its own stdout and stderr... so keep them (as we will
            # force stdout and stderr to several other things) so we can put
            # them back at the end
            nose_stdout = sys.stdout
            nose_stderr = sys.stderr
            sys.stdout = _SQWriter(sq, 'mock_stdout')
            sys.stderr = _SQWriter(sq, 'mock_stderr')
            self.assertNotIn(sys.stdout, (sys.__stdout__, nose_stdout))
            self.assertNotIn(sys.stderr, (sys.__stderr__, nose_stderr))

            sqw.MIN_READ_TIME = 'not a number'
            with self.assertRaises(TypeError):
                sqw.write('trigger an error')

            self.assertEqual(sys.stdout, sys.__stdout__)
            self.assertEqual(sys.stderr, sys.__stderr__)

            sys.stdout = nose_stdout
            sys.stderr = nose_stderr
            time.sleep(0.01)
            sq.get()


class ServerManagerTest(ServerManager):
    def _start_server(self):
        # don't really start the server - we'll test its pieces separately,
        # in the main process
        pass


class EmptyServer:
    def __init__(self, query_queue, response_queue, extras):
        query_queue.put('why?')
        response_queue.put(extras)


class CustomError(Exception):
    pass


def delayed_put(queue, val, delay):
    time.sleep(delay)
    queue.put(val)


class TestServerManager(TestCase):
    def check_error(self, manager, error_str, error_class):
        manager._response_queue.put(RESPONSE_ERROR, error_str)
        with self.assertRaises(error_class):
            manager.ask('which way does the wind blow?')

    def test_mechanics(self):
        extras = 'super secret don\'t tell anyone'

        sm = ServerManagerTest(name='test', server_class=EmptyServer,
                               shared_attrs=extras)
        sm._run_server()

        self.assertEqual(sm._query_queue.get(timeout=1), 'why?')
        self.assertEqual(sm._response_queue.get(timeout=1), extras)

        # builtin errors we propagate to the server
        builtin_error_str = ('traceback\n  lines\n and then\n'
                             '  OSError: your hard disk went floppy.')
        sm._response_queue.put((RESPONSE_ERROR, builtin_error_str))
        with self.assertRaises(OSError):
            sm.ask('which way does the wind blow?')

        # non-built-in errors we fall back on RuntimeError
        custom_error_str = ('traceback\nlines\nand then\n'
                            'CustomError: the Balrog is loose!')
        extra_resp1 = 'should get tossed by the error checker'
        extra_resp2 = 'so should this.'
        sm._response_queue.put((RESPONSE_OK, extra_resp1))
        sm._response_queue.put((RESPONSE_OK, extra_resp2))
        sm._response_queue.put((RESPONSE_ERROR, custom_error_str))

        # TODO: we have an intermittent failure below, but only when running
        # the full test suite (including pyqt and matplotlib?), not if we
        # run just this module, or at least not nearly as frequently.
        time.sleep(0.2)

        with LogCapture() as logs:
            with self.assertRaises(RuntimeError):
                sm.ask('something benign')
            self.assertTrue(sm._response_queue.empty())
        self.assertIn(extra_resp1, logs.value)
        self.assertIn(extra_resp2, logs.value)

        # extra responses to a query, only the last should be taken
        extra_resp1 = 'boo!'
        extra_resp2 = 'a barrel of monkeys!'
        sm._response_queue.put((RESPONSE_OK, extra_resp1))
        sm._response_queue.put((RESPONSE_OK, extra_resp2))
        time.sleep(0.05)
        p = mp.Process(target=delayed_put,
                       args=(sm._response_queue, (RESPONSE_OK, 42), 0.05))
        p.start()

        with LogCapture() as logs:
            self.assertEqual(sm.ask('what is the answer'), 42)
        self.assertIn(extra_resp1, logs.value)
        self.assertIn(extra_resp2, logs.value)

        # no response to a query
        with self.assertRaises(Empty):
            sm.ask('A sphincter says what?', timeout=0.05)

        # test halting an unresponsive server
        sm._server = mp.Process(target=time.sleep, args=(1000,))
        sm._server.start()

        self.assertIn(sm._server, mp.active_children())

        with LogCapture() as logs:
            sm.halt(0.01)
        self.assertIn('ServerManager did not respond '
                      'to halt signal, terminated', logs.value)

        self.assertNotIn(sm._server, mp.active_children())

    def test_pathological_edge_cases(self):
        # kill_queue should never fail
        kill_queue(None)

        # and halt should ignore AssertionErrors, which arise in
        # subprocesses when trying to kill a different subprocess
        sm = ServerManagerTest(name='test', server_class=None)

        class HorribleProcess:
            def is_alive(self):
                raise AssertionError

            def write(self):
                raise AssertionError

            def join(self):
                raise AssertionError

        sm._server = HorribleProcess()

        sm.halt()
