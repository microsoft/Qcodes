"""Unified qcodes test runners."""

import sys


def test_core(verbosity=1, failfast=False):
    """
    Run the qcodes core tests.

    Args:
        verbosity (int, optional): 0, 1, or 2, higher displays more info
            Default 1.
        failfast (bool, optional): If true, stops running on first failure
            Default False.

    Coverage testing is only available from the command line
    """
    import qcodes
    if qcodes.in_notebook():
        qcodes._IN_NOTEBOOK = True

    _test_core(verbosity=verbosity, failfast=failfast)


def _test_core(test_pattern='test*.py', **kwargs):
    import unittest

    import qcodes.tests as qctest
    import qcodes

    suite = unittest.defaultTestLoader.discover(
        qctest.__path__[0], top_level_dir=qcodes.__path__[0],
        pattern=test_pattern)
    if suite.countTestCases() == 0:
        print('found no tests')
        sys.exit(1)
    print('testing %d cases' % suite.countTestCases())

    result = unittest.TextTestRunner(**kwargs).run(suite)
    return result.wasSuccessful()


def test_part(name):
    """
    Run part of the qcodes core test suite.

    Args:
        name (str): a name within the qcodes.tests directory. May be:
            - a module ('test_loop')
            - a TestCase ('test_loop.TestLoop')
            - a test method ('test_loop.TestLoop.test_nesting')
    """
    import unittest
    fullname = 'qcodes.tests.' + name
    suite = unittest.defaultTestLoader.loadTestsFromName(fullname)
    return unittest.TextTestRunner().run(suite).wasSuccessful()

if __name__ == '__main__':
    import argparse
    import coverage
    import os
    import multiprocessing as mp
    import sys
    mp.set_start_method('spawn')

    # make sure coverage looks for .coveragerc in the right place
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description=('Core test suite for Qcodes, '
                     'covering everything except instrument drivers'))

    parser.add_argument('-v', '--verbose', nargs='?', dest='verbosity',
                        const=2, default=1, type=int,
                        help=('increase verbosity. default 1, '
                              '-v is the same as -v 2'))

    parser.add_argument('-c', '--coverage', nargs='?', dest='show_coverage',
                        const=1, default=1, type=int,
                        help=('show coverage. default is True '
                              '-c is the same as -c 1'))

    parser.add_argument('-t', '--test_pattern', nargs='?', dest='test_pattern',
                        const=1, default='test*.py', type=str,
                        help=('regexp for test name to match'))

    parser.add_argument('-f', '--failfast', nargs='?', dest='failfast',
                        const=1, default=0, type=int,
                        help=('halt on first error/failure? default 0 '
                              '(false), -f is the same as -f 1 (true)'))

    args = parser.parse_args()

    cov = coverage.Coverage()
    cov.start()

    success = _test_core(verbosity=args.verbosity,
                         failfast=bool(args.failfast),
                         test_pattern=args.test_pattern)

    cov.stop()
    # save coverage anyway since we computed it
    cov.save()
    if success and args.show_coverage:
        cov.report()
    # restore unix-y behavior
    # exit status 1 on fail
    if not success:
        sys.exit(1)
