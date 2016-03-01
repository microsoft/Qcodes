from unittest import TestCase, skipIf
import time
import re
import sys
import multiprocessing as mp
from unittest.mock import patch

import qcodes
from qcodes.utils.multiprocessing import (set_mp_method, QcodesProcess,
                                          get_stream_queue, _SQWriter)
import qcodes.utils.multiprocessing as qcmp
from qcodes.utils.helpers import in_notebook
from qcodes.utils.timing import calibrate

BREAK_SIGNAL = '~~BREAK~~'


class sqtest_echo:
    def __init__(self, name, delay=0.01, has_q=True):
        self.q_out = mp.Queue()
        self.q_err = mp.Queue()
        p = QcodesProcess(target=sqtest_echo_f,
                          args=(name, delay, self.q_out, self.q_err, has_q),
                          name=name)
        p.start()
        self.p = p
        self.delay = delay
        self.resp_delay = delay * 2 + 0.03

    def send_out(self, msg):
        self.q_out.put(msg)
        time.sleep(self.resp_delay)

    def send_err(self, msg):
        self.q_err.put(msg)
        time.sleep(self.resp_delay)

    def halt(self):
        if not self.p.is_alive():
            return
        self.q_out.put(BREAK_SIGNAL)
        self.p.join()
        time.sleep(self.resp_delay)

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
        mp_stats = calibrate()
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

    @patch('qcodes.utils.multiprocessing.in_notebook')
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

    @patch('qcodes.utils.multiprocessing.in_notebook')
    def test_qcodes_process(self, in_nb_patch):
        in_nb_patch.return_value = True

        queue_format = re.compile(
            '^\[\d\d:\d\d:\d\d\.\d\d\d p\d( ERR)?\] [^\[\]]*$')

        with self.sq.lock:
            p1 = sqtest_echo('p1')
            p2 = sqtest_echo('p2')
            time.sleep(self.MP_START_DELAY + p1.delay + p2.delay)

            self.assertEqual(self.sq.get(), '')

            procNames = ['<{}, started daemon>'.format(name)
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
                data = [line for line in self.sq.get().split('\n') if line]
                expected = [
                    label + 'row row row your boat',
                    label + 'gently down '
                ]
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


class TestSQWriter(TestCase):
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
