if __name__ == '__main__':
    import coverage
    import unittest
    import argparse

    parser = argparse.ArgumentParser(
        description='Static test suite for Qcodes')

    parser.add_argument('-v', '--verbose', nargs='?', dest='verbosity',
                        const=2, default=1, type=int,
                        help=('increase verbosity. default 1, '
                              '-v is the same as -v 2'))

    args = parser.parse_args()

    cov = coverage.Coverage()
    cov.start()

    import qcodes.tests as qctest

    suite = unittest.defaultTestLoader.discover(qctest.__path__[0])
    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)

    cov.stop()

    if result.wasSuccessful():
        cov.report()
