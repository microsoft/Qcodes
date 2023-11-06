# ruff: noqa: F401
"""
Module left for backwards compatibility. Will be deprecated and removed along the rest of qcodes.tests"""

from __future__ import annotations

import unittest

from qcodes.extensions import (
    DriverTestCase,
)
from qcodes.instrument import Instrument


def test_instruments(verbosity: int = 1) -> None:
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


def test_instrument(instrument_testcase, verbosity: int = 2) -> None:
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
