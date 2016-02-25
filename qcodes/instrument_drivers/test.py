import unittest

import qcodes.instrument_drivers as qcdrivers


class DriverTestCase(unittest.TestCase):
    driver = None  # override this in a subclass
    noskip = False  # just to test this class, we need to disallow skipping

    @classmethod
    def setUpClass(cls):
        if cls.driver is None:
            if cls is DriverTestCase:
                return
            else:
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
                  'testing the last one *****'.format(len(instances, name)))

        cls.instrument = instances[-1]


def test_instruments(verbosity=1):
    '''
    discover available instruments and test them all

    verbosity (default 2)
    '''
    driver_path = qcdrivers.__path__[0]
    suite = unittest.defaultTestLoader.discover(driver_path)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)
