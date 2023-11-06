# ruff: noqa: F401
"""
Module left for backwards compatibility. Will be deprecated and removed along the rest of qcodes.tests"""

from __future__ import annotations

import unittest

from qcodes.extensions import (
    DriverTestCase,
    test_instrument,
    test_instruments,
)
from qcodes.instrument import Instrument
