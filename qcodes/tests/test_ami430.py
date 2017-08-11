"""
A debug module for the AMI430 instrument driver. We cannot rely on the physical instrument to be present
which is why we need to mock it.
"""
import os
import sys

import numpy as np
import pytest

# Load the mock instrument servers
from qcodes.instrument.mock_ip import MockAMI430
# Load the instrument driver
from qcodes.instrument_drivers.american_magnetics.AMI430 import AMI430, AMI430_3D


@pytest.fixture(scope='module')
def current_driver(request):
    """
    Start three mock instruments representing current drivers for the x, y and z directions.
    """
    ip_address = "127.0.0.1"  # Should be local host
    ports = {"x": 1000, "y": 1001, "z": 1002}
    log_folder = None

    mock_instruments = []

    for axis, port in ports.items():

        if log_folder is not None:
            log_file = os.path.join(log_folder, "virtual_ip_log_{}.txt").format(axis)
        else:
            log_file = None

        mock_instrument = MockAMI430(axis, ip_address, port, log_file=log_file, silent=True)

        if mock_instrument.error == "Ok":
            mock_instrument.start()
            mock_instruments.append(mock_instrument)
        else:
            print(mock_instrument.error)
            sys.exit(1)

    def stop_mock_instruments():
        for mocker in mock_instruments:
            mocker.stop()

    request.addfinalizer(stop_mock_instruments)

    return instantiate_driver(ip_address, ports)


def instantiate_driver(ip_address, ports):

    coil_constant = 1  # [T/A]
    current_rating = 10  # [A]
    current_ramp_limit = 100  # [A/s]
    field_limit = 2  # [T]

    driver = AMI430_3D(
        "AMI430-3D",
        AMI430(
            "AMI430_x",
            ip_address,
            ports["x"],
            coil_constant,
            current_rating,
            current_ramp_limit,
            testing=True
        ),
        AMI430(
            "AMI430_y",
            ip_address,
            ports["y"],
            coil_constant,
            current_rating,
            current_ramp_limit,
            testing=True
        ),
        AMI430(
            "AMI430_z",
            ip_address,
            ports["z"],
            coil_constant,
            current_rating,
            current_ramp_limit,
            testing=True
        ),
        field_limit
    )

    return driver


def test_cylindrical_coordinates(current_driver):
    """
    We test a function call like "current_driver.cylindrical((phi, rho, z)" is equivalent to:

        current_driver.spherical((0.0, rho, z)
        current_driver.phi(phi)

    In fact, all of the following should be equivalent:

        current_driver.spherical((phi, 0.0, z)
        current_driver.rho(rho)

    and

        current_driver.spherical((phi, rho, 0.0)
        current_driver.z(z)

    The old version of the AMI430 driver did not display this equivalence. We test here if this bug has been solved.

    :param current_driver: The current driver to be tested
    """
    phi, rho, z = 30.0, 0.4, 0.5

    for count, name in enumerate(["phi", "rho", "z"]):

        args = [phi, rho, z]
        arg = args[count]
        args[count] = 0.0

        current_driver.cylindrical(tuple(args))
        getattr(current_driver, name)(arg)
        rho_m, phi_m, z_m = current_driver.cylindrical()
        # TODO: Notice how the input was given as "phi, rho, z". This inconsistency was already present in the
        # TODO: original driver in the repository. The fix is easy but we need to make sure backwards compatibility
        # TODO: is not lost so we do not break scripts already out there "in the wild"
        # TODO: --> should we change the input to "rho, phi, z" or the output to "phi, rho, z"?

        assert np.allclose([phi_m, rho_m, z_m], [phi, rho, z])


def test_spherical_coordinates(current_driver):
    """
    We test a function call like "current_driver.spherical((field, theta, phi)" is equivalent to:

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

    for count, name in enumerate(["field", "theta", "phi"]):

        args = [field, theta, phi]
        arg = args[count]
        args[count] = 0

        current_driver.spherical(args)
        getattr(current_driver, name)(arg)

        field_m, theta_m, phi_m = current_driver.spherical()
        assert np.allclose([field_m, theta_m, phi_m], [field, theta, phi])
