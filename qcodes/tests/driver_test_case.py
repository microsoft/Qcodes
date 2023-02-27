from __future__ import annotations

import unittest

from qcodes.instrument import Instrument

"""
This module defines:

- `DriverTestCase`: a `TestCase` subclass meant for testing instrument drivers

- `test_instrument`: a function to test one instrument, given its test class

- `test_instruments`: a function to test all instruments that have been defined
                      in your python session


Using `DriverTestCase` is pretty easy:

- Inherit from this class instead of from the base `unittest.TestCase`

- Provide a driver class variable that points to the Instrument class

- In your tests, `self.instrument` is the latest instance of this class.

- If your test case includes a `setUpClass` method, make sure to call
  `super().setUpClass()`, because that's where we find the latest instance of
  this `Instrument`, or skip the test case if no instances are found.
"""


class DriverTestCase(unittest.TestCase):
    # override this in a subclass
    driver: type[Instrument] | None = None
    instrument: Instrument

    @classmethod
    def setUpClass(cls):
        if cls is DriverTestCase:
            return

        if cls.driver is None:
            raise TypeError("you must set a driver for " + cls.__name__)

        instances = cls.driver.instances()
        name = cls.driver.__name__

        if not instances:
            msg = f"no instances of {name} found"
            if getattr(cls, "noskip", False):
                # just to test this class, we need to disallow skipping
                raise ValueError(msg)
            else:
                raise unittest.SkipTest(msg)

        if len(instances) == 1:
            print(f"***** found one {name}, testing *****")
        else:
            print(
                "***** found {} instances of {}; "
                "testing the last one *****".format(len(instances), name)
            )

        cls.instrument = instances[-1]


def test_instruments(verbosity=1) -> None:
    """
    Discover available instruments and test them all
    Unlike test_instrument, this does NOT reload tests prior to running them

    optional verbosity (default 1)
    """
    import qcodes
    import qcodes.instrument_drivers as qcdrivers

    driver_path = list(qcdrivers.__path__)[0]
    suite = unittest.defaultTestLoader.discover(
        driver_path, top_level_dir=list(qcodes.__path__)[0]
    )
    unittest.TextTestRunner(verbosity=verbosity).run(suite)


def test_instrument(instrument_testcase, verbosity=2) -> None:
    """
    Runs one instrument testcase
    Reloads the test case before running it

    optional verbosity (default 2)
    """
    import importlib
    import sys

    # reload the test case
    module_name = instrument_testcase.__module__
    class_name = instrument_testcase.__name__
    del sys.modules[module_name]

    module = importlib.import_module(module_name)
    reloaded_testcase = getattr(module, class_name)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(reloaded_testcase)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)
