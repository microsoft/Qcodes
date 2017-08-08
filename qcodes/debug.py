"""
A debug module for the AMI430 instrument driver. We cannot rely on the physical instrument to be present
which is why we need to mock it.
"""
import os
import sys
import traceback
import numpy as np

# Load the mock instrument servers
from qcodes.instrument.mock_ip import MockAMI430
# Load the instrument driver
from qcodes.instrument_drivers.american_magnetics.AMI430 import AMI430, AMI430_3D


def start_mock_instruments(ip_address, ports):
    """
    Start three mock instruments representing current drivers for the x, y and z directions.
    """

    log_file = os.path.join("C:\\", "Users", "a-sochat", "Desktop", "virtual_ip_log_{}.txt")

    mock_instruments = []

    for axis, port in ports.items():

        mock_instrument = MockAMI430(axis, ip_address, port, log_file.format(axis), silent=True)

        if mock_instrument.error == "Ok":
            mock_instrument.start()
            mock_instruments.append(mock_instrument)
        else:
            print(mock_instrument.error)
            sys.exit(1)

    return mock_instruments


def spherical_to_cartesian(r, theta, phi):

    phi, theta = np.radians(phi), np.radians(theta)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z


def main():

    ip_address = "127.0.0.1"
    ports = {"x": 1000, "y": 1001, "z": 1002}

    mock_instruments = start_mock_instruments(ip_address, ports)

    coil_constant = 1  # [T/A]
    current_rating = 10  # [A]
    current_ramp_limit = 100  # [A/s]
    field_limit = 2  # [T]

    current_driver = AMI430_3D(
        "AMI430-3D",
        AMI430(
            "AMI430_x",
            ip_address,
            ports["x"],
            coil_constant,
            current_rating,
            current_ramp_limit
        ),
        AMI430(
            "AMI430_y",
            ip_address,
            ports["y"],
            coil_constant,
            current_rating,
            current_ramp_limit
        ),
        AMI430(
            "AMI430_z",
            ip_address,
            ports["z"],
            coil_constant,
            current_rating,
            current_ramp_limit
        ),
        field_limit
    )

    # Lets try to do something useful with the driver...
    try:
        field, theta, phi = 0.4, 3.0, 10.0
        current_driver._set_spherical((field, theta, phi))

        print(spherical_to_cartesian(field, theta, phi))
        print(current_driver._get_measured(*("x", "y", "z")))


    except:
        traceback.print_exc()
    finally:
        # Stop the mock instruments
        for mocker in mock_instruments:
            mocker.stop()

if __name__ == "__main__":
    main()