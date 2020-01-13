import io
import pytest

from qcodes.instrument_drivers.Lakeshore.Model_325 import read_curve_file, get_sanitize_data

curve_file_content = """ \
Sensor Model:   CX-1050-SD-HT-1.4L
Serial Number:  X116121
Interpolation Method:   Lagrangian
SetPoint Limit: 325.0      (Kelvin)
Data Format:    4      (Log Ohms/Kelvin)
Number of Breakpoints:   52

No.   Units      Temperature (K)

  1  1.70333       325.0
  2  1.70444       324.0
  3  1.72168       309.0
  4  1.73995       294.0
  5  1.75936       279.0
  6  1.78000       264.0
"""


@pytest.fixture(scope="function")
def curve_file():
    yield io.StringIO(curve_file_content)


def test_file_parser(curve_file):
    file_data = read_curve_file(curve_file)

    assert list(file_data.keys()) == ["metadata", "data"]

    assert file_data["metadata"] == {
        "Sensor Model": "CX-1050-SD-HT-1.4L",
        "Serial Number":  "X116121",
        "Interpolation Method": "Lagrangian",
        "SetPoint Limit": "325.0      (Kelvin)",
        "Data Format": "4      (Log Ohms/Kelvin)",
        "Number of Breakpoints": "52"
    }

    assert file_data["data"] == {
        "No.": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        "Units": (1.70333, 1.70444, 1.72168, 1.73995, 1.75936, 1.78),
        "Temperature (K)": (325.0, 324.0, 309.0, 294.0, 279.0, 264.0)
    }


def test_sanitise_data(curve_file):

    file_data = read_curve_file(curve_file)
    data_dict = get_sanitize_data(file_data)

    assert data_dict == {
        "log Ohm": (1.70333, 1.70444, 1.72168, 1.73995, 1.75936, 1.78),
        "Temperature (K)": (325.0, 324.0, 309.0, 294.0, 279.0, 264.0)
    }
