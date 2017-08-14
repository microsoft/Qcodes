"""
A debug module for the AMI430 instrument driver. We cannot rely on the physical instrument to be present
which is why we need to mock it.
"""
import os
import re
import sys
from queue import Queue

import numpy as np
import pytest

# Load the mock instrument servers
from qcodes.instrument.mock_ip import MockAMI430
# Load the instrument driver
from qcodes.instrument_drivers.american_magnetics.AMI430 import AMI430, AMI430_3D
from qcodes.math.field_vector import FieldVector

coil_constant = 1  # [T/A]
current_rating = 10  # [A]
current_ramp_limit = 100  # [A/s]
field_limit = 2  # [T]


class StdOutQueue(Queue):
    """
    This class will allow us to redirect stdout to a Queue, which can be handy to test if correct output is
    printed to screen. This is also handy for inter-thread communication.
    """

    def __init__(self, *args, **kwargs):
        Queue.__init__(self, *args, **kwargs)

    def write(self, msg):
        self.put(msg)

    def flush(self):
        sys.__stdout__.flush()


msg_stream = StdOutQueue()  # Mock instrument output shall be available through this Queue


@pytest.fixture(scope='module')
def current_driver(request):
    """
    Start three mock instruments representing current drivers for the x, y and z directions.
    """
    ip_address = "127.0.0.1"  # Should be local host
    ports = {"x": 1025, "y": 1026, "z": 1027}  # Ports lower then 1024 are reserved under linux
    log_folder = None

    mock_instruments = []

    for axis, port in ports.items():

        if log_folder is not None:
            log_file = os.path.join(log_folder, "virtual_ip_log_{}.txt").format(axis)
        else:
            log_file = None

        mock_instrument = MockAMI430(axis, ip_address, port, log_file=log_file, output_stream=msg_stream)

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


def get_output_msg():
    """
    Get output messages from the mock instruments. These are normally written to stdout, but are now redirected
    to msg_stream
    """
    messages = []
    while True:
        try:
            msg = msg_stream.get_nowait()
            messages.append(msg)
        except:  # The queue is empty
            break

    return messages


def get_reported_ramp_targets():
    """
    Listen to the mock instruments and parse the messages to extract the ramping targets. This is useful in determining
    if the set targets arrive at the individual instruments correctly.
    """

    search_string = "([x, y, z]): Ramping to (.*)"
    reported_ramp_targets = {}
    messages = get_output_msg()

    for msg in messages:
        result = re.search(search_string, msg)
        if result is not None:
            device_name, value = result.groups()
            reported_ramp_targets[device_name] = float(value)

    return reported_ramp_targets


def get_random_coordinate(coordinate_name):
    return {
        "x": np.random.uniform(-1, 1),
        "y": np.random.uniform(-1, 1),
        "z": np.random.uniform(-1, 1),
        "r": np.random.uniform(0, 2),
        "theta": np.random.uniform(0, 180),
        "phi": np.random.uniform(0, 360),
        "rho": np.random.uniform(0, 1)
    }[coordinate_name]


def test_cartesian_sanity(current_driver):
    n_repeats = 10

    for _ in range(n_repeats):
        set_target = [get_random_coordinate(name) for name in ["x", "y", "z"]]
        current_driver.cartesian(set_target)
        get_target = current_driver.cartesian()

        assert np.allclose(set_target, get_target)


def test_spherical_sanity(current_driver):
    n_repeats = 10

    for _ in range(n_repeats):
        set_target = [get_random_coordinate(name) for name in ["r", "theta", "phi"]]
        current_driver.spherical(set_target)
        get_target = current_driver.spherical()

        assert np.allclose(set_target, get_target)


def test_cylindrical_sanity(current_driver):
    n_repeats = 10

    for _ in range(n_repeats):
        set_target = [get_random_coordinate(name) for name in ["rho", "phi", "z"]]
        current_driver.spherical(set_target)
        get_target = current_driver.spherical()

        assert np.allclose(set_target, get_target)


def test_cartesian_setpoints(current_driver):
    n_repeats = 10

    for _ in range(n_repeats):
        set_target = [get_random_coordinate(name) for name in ["x", "y", "z"]]
        current_driver.cartesian(set_target)

        get_target = get_reported_ramp_targets()
        # The mock instruments talk to us. Normally these messages are directed
        # to stdout but in this test suite they are directed to "msg_stream". Extract what the instruments are saying
        # about which fields are being ramped to and verify if this matches with the set targets.

        set_vector = FieldVector(*set_target)
        get_vector = FieldVector(**get_target)
        assert set_vector.is_equal(get_vector)


def test_spherical_setpoints(current_driver):
    n_repeats = 10
    names = ["r", "theta", "phi"]

    for _ in range(n_repeats):
        set_target = {name: get_random_coordinate(name) for name in names}
        current_driver.spherical([set_target[name] for name in names])

        get_target = get_reported_ramp_targets()

        set_vector = FieldVector(**set_target)
        get_vector = FieldVector(**get_target)
        assert set_vector.is_equal(get_vector)


def test_cylindrical_setpoints(current_driver):
    n_repeats = 10
    names = ["rho", "phi", "z"]

    for _ in range(n_repeats):
        set_target = {name: get_random_coordinate(name) for name in names}
        current_driver.cylindrical([set_target[name] for name in names])

        get_target = get_reported_ramp_targets()

        set_vector = FieldVector(**set_target)
        get_vector = FieldVector(**get_target)
        assert set_vector.is_equal(get_vector)


def test_cylindrical_poles(current_driver):
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
    rho, phi, z = 0.4, 30.0, 0.5

    for count, name in enumerate(["rho", "phi", "z"]):
        args = [rho, phi, z]
        arg = args[count]
        args[count] = 0.0

        current_driver.cylindrical(tuple(args))
        getattr(current_driver, name)(arg)
        rho_m, phi_m, z_m = current_driver.cylindrical()

        assert np.allclose([phi_m, rho_m, z_m], [phi, rho, z])


def test_spherical_poles(current_driver):
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


def test_field_limit_exception(current_driver):
    """
    Test that an exception is raised if we intentionally set the field beyond the limits
    """
    x, y, z = field_limit + 0.1, 0, 0

    with pytest.raises(Exception) as excinfo:
        current_driver.cartesian((x, y, z))

    assert "field would exceed limit" in excinfo.value.args[0]
