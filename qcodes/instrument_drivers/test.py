import unittest


class DriverTestCase(unittest.TestCase):
    driver = None  # override this in a subclass
    noskip = False  # just to test this class, we need to disallow skipping

    @classmethod
    def setUpClass(cls):
        if cls is DriverTestCase:
            return

        if cls.driver is None:
            raise TypeError('you must set a driver for ' + cls.__name__)

        instances = cls.driver.instances()
        name = cls.driver.__name__

        if not instances:
            msg = 'no instances of {} found'.format(name)
            if cls.noskip:
                raise ValueError(msg)
            else:
                raise unittest.SkipTest(msg)

        if len(instances) == 1:
            print('***** found one {}, testing *****'.format(name))
        else:
            print('***** found {} instances of {}; '
                  'testing the last one *****'.format(len(instances), name))

        cls.instrument = instances[-1]


def test_instruments(verbosity=1):
    '''
    Discover available instruments and test them all
    Unlike test_instrument, this does NOT reload tests prior to running them

    optional verbosity (default 1)
    '''
    import qcodes.instrument_drivers as qcdrivers
    import qcodes

    driver_path = qcdrivers.__path__[0]
    suite = unittest.defaultTestLoader.discover(
        driver_path, top_level_dir=qcodes.__path__[0])
    unittest.TextTestRunner(verbosity=verbosity).run(suite)


def test_instrument(instrument_testcase, verbosity=2):
    '''
    Runs one instrument testcase
    Reloads the test case before running it

    optional verbosity (default 2)
    '''
    import sys
    import importlib

    # reload the test case
    module_name = instrument_testcase.__module__
    class_name = instrument_testcase.__name__
    del sys.modules[module_name]

    module = importlib.import_module(module_name)
    reloaded_testcase = getattr(module, class_name)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(reloaded_testcase)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)
