from qcodes.utils import issue_deprecation_warning

issue_deprecation_warning(
    "`qcodes.tests` module",
    "tests are no longer shipped with QCoDeS",
    "`qcodes.instrument_drivers.mock_instruments` and `qcodes.extensions.DriverTestCase`",
)
