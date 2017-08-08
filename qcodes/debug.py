"""
A debug module for the AMI430 instrument driver. We cannot rely on the physical instrument to be present
which is why we need to mock it.
"""
import os
import sys
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


def instantiate_driver(ip_address, ports):

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

    return current_driver


def test_cylindrical_coordinates(current_driver):

    phi, rho, z = 30.0, 0.4, 0.5

    for count, set_function in enumerate(
            [current_driver._set_phi, current_driver._set_rho, current_driver._set_z]):

        args = [phi, rho, z]
        #arg = args[count]
        #args[count] = 0

        current_driver._set_cylindrical(tuple(args))
        #set_function(arg)

        phi_m, rho_m, z_m = current_driver._get_measured("phi", "rho", "z")
        phi_m = np.degrees(phi_m)

        assert np.allclose([phi_m, rho_m, z_m], [phi, rho, z])


def test_spherical_coordinates(current_driver):
    """
    We test a function call like "current_driver.set_spherical((field, theta, phi)" is equivalent to:

        current_driver.set_spherical((0.0, theta, phi)
        current_driver.set_field(field)

    In fact, all of the following should be equivalent:

        current_driver.set_spherical((field, 0.0, phi)
        current_driver.set_theta(theta)

    and

        current_driver.set_spherical((field, theta, 0.0)
        current_driver.set_phi(phi)

    The old version of the AMI430 driver did not display this equivalence. We test here if this bug has been solved.

    :param current_driver: The current driver to be tested
    """

    field, theta, phi = 0.5, 30.0, 50.0

    for count, set_function in enumerate(
            [current_driver._set_field, current_driver._set_theta, current_driver._set_phi]):

        args = [field, theta, phi]
        arg = args[count]
        args[count] = 0

        current_driver._set_spherical(tuple(args))
        set_function(arg)

        field_m, theta_m, phi_m = current_driver._get_measured("field", "theta", "phi")
        theta_m = np.degrees(theta_m)
        phi_m = np.degrees(phi_m)

        assert np.allclose([field_m, theta_m, phi_m], [field, theta, phi])


def main():
    ip_address = "127.0.0.1"
    ports = {"x": 1000, "y": 1001, "z": 1002}

    mock_instruments = start_mock_instruments(ip_address, ports)
    current_driver = instantiate_driver(ip_address, ports)

    #test_spherical_coordinates(current_driver)
    test_cylindrical_coordinates(current_driver)

    for mocker in mock_instruments:
        mocker.stop()

if __name__ == "__main__":
    main()