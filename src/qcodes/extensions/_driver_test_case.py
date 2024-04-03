from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qcodes.instrument import Instrument

"""
This module defines:

- `DriverTestCase`: a `TestCase` subclass meant for testing instrument drivers

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
    def setUpClass(cls) -> None:
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
                f"***** found {len(instances)} instances of {name}; "
                "testing the last one *****"
            )

        cls.instrument = instances[-1]
