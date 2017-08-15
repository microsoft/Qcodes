"""
A debug module for the AMI430 instrument driver. We cannot rely on the physical instrument to be present
which is why we need to mock it.
"""
import os
import re
import sys
from datetime import datetime
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
        # Alternatively, if output_stream="stdout", messages will be printed to screen.

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


def get_instrument_logs():
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


def get_instruments_ramp_messages():
    """
    Listen to the mock instruments and parse the messages to extract the ramping targets. This is useful in determining
    if the set targets arrive at the individual instruments correctly. The time stamps are useful to test that the
    messages arrive in the correct order.
    """

    search_string = "\[(.*)\] ([x, y, z]): Ramping to (.*)"
    # We expect the log messages to be in the format "[<time stamp>] <name>: <message>", where name is either x, y or z
    reported_ramp_targets = {}
    messages = get_instrument_logs()

    for msg in messages:
        result = re.search(search_string, msg)
        if result is not None:
            time_string, device_name, value = result.groups()
            time_value = datetime.strptime(time_string, "%d:%m:%Y-%H:%M:%S.%f")
            reported_ramp_targets[device_name] = {"value": float(value), "time": time_value}

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
    """
    A sanity check to see if the driver remember vectors in any random configuration in cartesian coordinates
    """
    n_repeats = 10

    for _ in range(n_repeats):
        set_target = [get_random_coordinate(name) for name in ["x", "y", "z"]]
        current_driver.cartesian(set_target)
        get_target = current_driver.cartesian()

        assert np.allclose(set_target, get_target)


def test_spherical_sanity(current_driver):
    """
    A sanity check to see if the driver remember vectors in any random configuration in spherical coordinates
    """
    n_repeats = 10

    for _ in range(n_repeats):
        set_target = [get_random_coordinate(name) for name in ["r", "theta", "phi"]]
        current_driver.spherical(set_target)
        get_target = current_driver.spherical()

        assert np.allclose(set_target, get_target)


def test_cylindrical_sanity(current_driver):
    """
    A sanity check to see if the driver remember vectors in any random configuration in cylindrical coordinates
    """
    n_repeats = 10

    for _ in range(n_repeats):
        set_target = [get_random_coordinate(name) for name in ["rho", "phi", "z"]]
        current_driver.cylindrical(set_target)
        get_target = current_driver.cylindrical()

        assert np.allclose(set_target, get_target)


def test_cartesian_setpoints(current_driver):
    """
    Check that the individual x, y, z instruments are getting the set points as intended. We can do this because the
    instruments are printing log messages to an IO stream. We intercept these messages and extract the log lines which
    mention that the instrument is ramping to certain values. These values should match the values of the input. In
    this test we are verifying this for cartesian coordinates.
    """
    n_repeats = 10

    for _ in range(n_repeats):
        set_target = [get_random_coordinate(name) for name in ["x", "y", "z"]]
        current_driver.cartesian(set_target)

        reported_ramp_targets = get_instruments_ramp_messages()
        get_target = {k: v["value"] for k, v in reported_ramp_targets.items()}

        set_vector = FieldVector(*set_target)
        get_vector = FieldVector(**get_target)
        assert set_vector.is_equal(get_vector)


def test_spherical_setpoints(current_driver):
    """
    Check that the individual x, y, z instruments are getting the set points as intended. We can do this because the
    instruments are printing log messages to an IO stream. We intercept these messages and extract the log lines which
    mention that the instrument is ramping to certain values. These values should match the values of the input. In
    this test we are verifying this for spherical coordinates.
    """
    n_repeats = 10
    names = ["r", "theta", "phi"]

    for _ in range(n_repeats):
        set_target = {name: get_random_coordinate(name) for name in names}
        current_driver.spherical([set_target[name] for name in names])

        reported_ramp_targets = get_instruments_ramp_messages()
        get_target = {k: v["value"] for k, v in reported_ramp_targets.items()}

        set_vector = FieldVector(**set_target)
        get_vector = FieldVector(**get_target)
        assert set_vector.is_equal(get_vector)


def test_cylindrical_setpoints(current_driver):
    """
    Check that the individual x, y, z instruments are getting the set points as intended. We can do this because the
    instruments are printing log messages to an IO stream. We intercept these messages and extract the log lines which
    mention that the instrument is ramping to certain values. These values should match the values of the input. In
    this test we are verifying this for cylindrical coordinates.
    """
    n_repeats = 10
    names = ["rho", "phi", "z"]

    for _ in range(n_repeats):
        set_target = {name: get_random_coordinate(name) for name in names}
        current_driver.cylindrical([set_target[name] for name in names])

        reported_ramp_targets = get_instruments_ramp_messages()
        get_target = {k: v["value"] for k, v in reported_ramp_targets.items()}

        set_vector = FieldVector(**set_target)
        get_vector = FieldVector(**get_target)
        assert set_vector.is_equal(get_vector)


def test_ramp_down_first(current_driver):
    """
    To prevent quenching of the magnets, we need the driver to always be within the field limits. Part of the strategy
    of making sure that this is the case includes always performing ramp downs first. For example, if magnets x and z
    need to be ramped up, while magnet y needs to be ramped down, ramp down magnet y first and then ramp up x and z.
    This is tested here.
    """
    names = ["x", "y", "z"]
    set_point = np.array([0.3, 0.4, 0.5])
    current_driver.cartesian(set_point)  # Put the magnets in a well defined state
    delta = np.array([-0.1, 0.1, 0.1])  # We begin with ramping down x first while ramping up y and z

    for count, ramp_down_name in enumerate(names):
        set_point += np.roll(delta, count)  # The second iteration will ramp down y while ramping up x an z.
        # Check if y is adjusted first. We will perform the same test with z in the third iteration

        current_driver.cartesian(set_point)
        reported_ramp_targets = get_instruments_ramp_messages()  # get the logging outputs from the instruments.
        times = {k: v["time"] for k, v in reported_ramp_targets.items()}  # extract the time stamps

        for ramp_up_name in np.delete(names, count):
            assert times[ramp_down_name] < times[ramp_up_name]  # ramp down occurs before ramp up.


def test_field_limit_exception(current_driver):
    """
    Test that an exception is raised if we intentionally set the field beyond the limits. Together with the
    test_ramp_down_first test this should prevent us from ever exceeding set point limits.
    """
    names = ["x", "y", "z"]
    set_point = np.array([0.1, 0.1, 0.1])
    current_driver.cartesian(set_point)  # Put the magnets in a well defined state
    delta = np.array([field_limit, 0.0, 0.0])

    for count, ramp_down_name in enumerate(names):
        set_point += np.roll(delta, count)

        with pytest.raises(Exception) as excinfo:
            current_driver.cartesian(set_point)

        assert "field would exceed limit" in excinfo.value.args[0]


def test_cylindrical_poles(current_driver):
    """
    Test that the phi coordinate is remembered even if the resulting vector is equivalent to the null vector
    """
    rho, phi, z = 0.4, 30.0, 0.5
    current_driver.cylindrical((0.0, phi, 0.0))  # This is equivalent to the null vector
    current_driver.rho(rho)
    current_driver.z(z)

    rho_m, phi_m, z_m = current_driver.cylindrical()  # after setting the rho and z values we should have the vector as
    # originally intended.
    assert np.allclose([rho_m, phi_m, z_m], [rho, phi, z])


def test_spherical_poles(current_driver):
    """
    Test that the theta and phi coordinates are remembered even if the resulting vector is equivalent to the null vector
    """

    field, theta, phi = 0.5, 30.0, 50.0
    current_driver.spherical((0.0, theta, phi))  # If field=0, then this is equivalent to the null vector
    # x, y, z = (0, 0, 0)
    current_driver.field(field)

    field_m, theta_m, phi_m = current_driver.spherical()  # after setting the field we should have the vector as
    # originally intended.
    assert np.allclose([field_m, theta_m, phi_m], [field, theta, phi])
