from unittest import TestCase
import time
import re
import sys
import multiprocessing as mp

from qcodes.utils.multiprocessing import (set_mp_method, QcodesProcess,
                                          get_stream_queue)


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
            print('error from {}...'.format(name), end='',
                  file=sys.stderr, flush=True)
            print('', end='')  # this one should do nothing
        time.sleep(period)
    print('')  # this one should make a blank line at the very end

    # now test that disconnect works, and reverts to regular stdout and stderr
    get_stream_queue().disconnect()
    print('stdout ', end='', flush=True)
    print('stderr ', file=sys.stderr, end='', flush=True)


class TestMpMethod(TestCase):
    pass
    # TODO - this is going to be a bit fragile and platform-dependent, I think.
    # we will need to initialize the start method before *any* tests run,
    # which looks like it will require using a plugin.


class TestQcodesProcess(TestCase):
    def test_qcodes_process(self):
        # set up two processes that take 0.5 and 0.4 sec and produce
        # staggered results
        # p1 produces more output and sometimes makes two messages in a row
        # before p1 produces any.
        sq = get_stream_queue()
        queue_format = re.compile(
            '^\[\d\d:\d\d:\d\d\.\d\d\d p\d( ERR)?\] [^\[\]]*$')

        sqtest('p1', 0.05, 10)
        time.sleep(0.025)
        sqtest('p2', 0.1, 4)

        reprs = [repr(p) for p in mp.active_children()]
        for name in ('p1', 'p2'):
            self.assertIn('<{}, started daemon>'.format(name), reprs)
        self.assertEqual(len(reprs), 2, reprs)

        time.sleep(0.25)
        queue_data1 = sq.get().split('\n')

        time.sleep(0.25)
        self.assertEqual(mp.active_children(), [])
        queue_data2 = sq.get().split('\n')

        for line in queue_data1 + queue_data2[1:-1]:
            self.assertIsNotNone(queue_format.match(line), line)
        # we've tested the header, now strip it
        data1 = [line[14:] for line in queue_data1]
        data2 = [line[14:] for line in queue_data2[1:-1]]
        p1msg = 'p1] message from P1...'
        p2msg = p1msg.replace('1', '2')
        p1err = 'p1 ERR] error from P1...'
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
