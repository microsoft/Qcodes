from unittest import TestCase
import time
import re
import sys
import multiprocessing as mp
from unittest.mock import patch

from qcodes.utils.multiprocessing import (set_mp_method, QcodesProcess,
                                          get_stream_queue, _SQWriter)
import qcodes.utils.multiprocessing as qcmp
from qcodes.utils.helpers import in_notebook
from qcodes.utils.timing import calibrate


# note sometimes separate processes do not seem to register in
# coverage tests, though it seems that when we actively return stdout and
# stderr to the normal routes, coverage does get tested?
# if that starts to fail, we can insert "pragma no cover" comments around
# such code - but need to be extra careful then that we really do cover it!

def sqtest(name, period, cnt):
    p = QcodesProcess(target=sqtest_f, args=(name.upper(), period, cnt),
                      name=name)
    p.start()
    return p


def sqtest_f(name, period, cnt):
    for i in range(cnt):
        print('message from {}...'.format(name), end='', flush=True)
        if i % 5 == 1:
            print('mock error from {}...'.format(name), end='',
                  file=sys.stderr, flush=True)
            print('', end='')  # this one should do nothing
        time.sleep(period)
    print('')  # this one should make a blank line at the very end

    # now test that disconnect works, and reverts to regular stdout and stderr
    get_stream_queue().disconnect()
    print('stdout ', end='', flush=True)
    print('stderr ', file=sys.stderr, end='', flush=True)


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

    def test_not_in_notebook(self):
        # below we'll patch this to True, but make sure that it's False
        # in the normal test runner.
        self.assertEqual(in_notebook(), False)

        # and make sure that processes run this way do not use the queue
        period = 0.01
        cnt = 4
        sq = get_stream_queue()
        with sq.lock:
            p = sqtest('p0', period, cnt)
            self.assertIsNone(p.stream_queue)
            time.sleep(self.MP_START_DELAY + self.MP_FINISH_DELAY +
                       cnt * (period + self.SLEEP_DELAY) + 0.05)

            self.assertEqual(sq.get(), '')

    @patch('qcodes.utils.multiprocessing.in_notebook')
    def test_qcodes_process(self, in_nb_patch):
        in_nb_patch.return_value = True

        # set up two processes that produce staggered results
        # p1 produces more output and sometimes makes two messages in a row
        # before p1 produces any.
        sq = get_stream_queue()
        queue_format = re.compile(
            '^\[\d\d:\d\d:\d\d\.\d\d\d p\d( ERR)?\] [^\[\]]*$')

        with sq.lock:
            # the whole thing takes ~ base_period * 10 seconds
            # base_period used to be 0.05, but we increased for
            # robustness - especially with coverage on the timing can
            # get a bit off what it's supposed to be
            base_period = 0.1
            sqtest('p1', base_period, 10)
            time.sleep(base_period / 2)
            sqtest('p2', 2 * base_period, 4)
            time.sleep(self.MP_START_DELAY + 5 * base_period)

            procNames = ['<{}, started daemon>'.format(name)
                         for name in ('p1', 'p2')]

            reprs = [repr(p) for p in mp.active_children()]
            for name in procNames:
                self.assertIn(name, reprs)

            # Some OS's start more processes just for fun... so don't test
            # that p1 and p2 are the only ones.
            # self.assertEqual(len(reprs), 2, reprs)

            queue_data1 = sq.get().split('\n')

            time.sleep(5 * base_period + 15 * self.SLEEP_DELAY +
                       4 * self.MP_FINISH_DELAY)

            # both p1 and p2 should have finished by now, and ended.
            reprs = [repr(p) for p in mp.active_children()]
            for name in procNames:
                self.assertNotIn(name, reprs)

        queue_data2 = sq.get().split('\n')

        if queue_data1[0] == '':
            # sometimes we get a blank here - not sure why...
            # but it wouldn't cause any problems in the real queue anyway
            queue_data1 = queue_data1[1:]

        real_lines = queue_data1 + queue_data2[1:-1]
        for line in real_lines:
            self.assertIsNotNone(queue_format.match(line), real_lines)
        # we've tested the header, now strip it
        data1 = [line[14:] for line in queue_data1]
        data2 = [line[14:] for line in queue_data2[1:-1]]
        p1msg = 'p1] message from P1...'
        p2msg = p1msg.replace('1', '2')
        p1err = 'p1 ERR] mock error from P1...'
        p2err = p1err.replace('1', '2')
        expected_data1 = [
            p1msg,
            p2msg,
            p1msg,
            p1err,
            p1msg,
            p2msg,
            p2err,
            p1msg + p1msg[4:],
            p2msg,
            p1msg
        ]
        # first line of data2 is special, as it has no header
        # because it's continuing a line from the same stream
        expected_data2_first = p1msg[4:]
        expected_data2 = [
            p1err,
            p2msg,
            p1msg + p1msg[4:],
            'p2] ',  # p2 is quitting - should send a blank line
            p1msg
        ]
        self.assertEqual(data1, expected_data1)
        self.assertEqual(queue_data2[0], expected_data2_first)
        self.assertEqual(data2, expected_data2)
        # last line of data2 is also special, it's a trailing blank
        # when p1 quits
        self.assertEqual(queue_data2[-1], '')


class TestSQWriter(TestCase):
    # this is basically tested in TestQcodesProcess, but the test happens
    # in a subprocess so coverage doesn't know about it. Anyway, there are
    # a few edge cases left that we have to test locally.
    # @patch('qcodes.utils.multiprocessing.sys.__stdout__')
    def test_sq_writer(self):  # , base_stdout_patch):
        # import pdb; pdb.set_trace()
        sq = get_stream_queue()
        with sq.lock:
            sq_clearer = _SQWriter(sq, 'Someone else')
            sq_clearer.write('Boo!\n')

            # apparently we need a little delay to make sure the queue
            # properly appears populated.
            time.sleep(0.01)

            sq.get()
            sq_name = 'Magritte'
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
            new_message = 'Who?\n'
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
