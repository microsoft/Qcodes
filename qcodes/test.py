def test_core(verbosity=1):
    '''
    Run the qcodes core tests.

    Coverage testing is only available from the command line
    '''
    import qcodes
    if qcodes.in_notebook():
        qcodes._IN_NOTEBOOK = True

    _test_core(verbosity=verbosity)


def _test_core(verbosity):
    import unittest

    import qcodes.tests as qctest
    import qcodes

    suite = unittest.defaultTestLoader.discover(
        qctest.__path__[0], top_level_dir=qcodes.__path__[0])
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return result.wasSuccessful()

if __name__ == '__main__':
    import argparse
    import coverage
    import os

    # make sure coverage looks for .coveragerc in the right place
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description=('Core test suite for Qcodes, '
                     'covering everything except instrument drivers'))

    parser.add_argument('-v', '--verbose', nargs='?', dest='verbosity',
                        const=2, default=1, type=int,
                        help=('increase verbosity. default 1, '
                              '-v is the same as -v 2'))

    args = parser.parse_args()

    cov = coverage.Coverage()
    cov.start()

    success = _test_core(verbosity=args.verbosity)

    cov.stop()

    if success:
        cov.report()
