import io
import logging
import re
import time
import warnings
from contextlib import ExitStack
from typing import Any, TypedDict

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis.strategies import floats, tuples
from pytest import FixtureRequest, LogCaptureFixture

from qcodes.instrument import Instrument
from qcodes.instrument_drivers.american_magnetics import (
    AMI430Warning,
    AMIModel430,
    AMIModel4303D,
)
from qcodes.instrument_drivers.american_magnetics.AMI430_visa import AMI430, AMI430_3D
from qcodes.math_utils import FieldVector
from qcodes.utils.types import (
    numpy_concrete_floats,
    numpy_concrete_ints,
    numpy_non_concrete_ints_instantiable,
)

_time_resolution = time.get_clock_info("time").resolution

# If any of the field limit functions are satisfied we are in the safe zone.
# We can have higher field along the z-axis if x and y are zero.
field_limit = [
    lambda x, y, z: x == 0 and y == 0 and z < 3,
    lambda x, y, z: np.linalg.norm([x, y, z]) < 2,
]

LOG_NAME = "qcodes.instrument.instrument_base"


@pytest.fixture(scope="function")
def magnet_axes_instances():
    """
    Start three mock instruments representing current drivers for the x, y,
    and z directions.
    """
    mag_x = AMIModel430(
        "x", address="GPIB::1::INSTR", pyvisa_sim_file="AMI430.yaml", terminator="\n"
    )
    mag_y = AMIModel430(
        "y", address="GPIB::2::INSTR", pyvisa_sim_file="AMI430.yaml", terminator="\n"
    )
    mag_z = AMIModel430(
        "z", address="GPIB::3::INSTR", pyvisa_sim_file="AMI430.yaml", terminator="\n"
    )

    yield mag_x, mag_y, mag_z

    mag_x.close()
    mag_y.close()
    mag_z.close()


@pytest.fixture(name="current_driver", scope="function")
def _make_current_driver(magnet_axes_instances):
    """
    Instantiate AMI430_3D instrument with the three mock instruments
    representing current drivers for the x, y, and z directions.
    """
    mag_x, mag_y, mag_z = magnet_axes_instances

    driver = AMIModel4303D("AMI430_3D", mag_x, mag_y, mag_z, field_limit)

    yield driver

    driver.close()


@pytest.fixture(scope="function", name="ami430")
def _make_ami430():
    mag = AMIModel430(
        "ami430",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="AMI430.yaml",
        terminator="\n",
    )
    yield mag
    mag.close()


# here the original test has a homemade log system that we don't want to
# reproduce / write tests for. Instead, we use normal logging from our
# instrument.visa module
iostream = io.StringIO()
logger = logging.getLogger("qcodes.instrument.visa")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(created)s - %(message)s")
lh = logging.StreamHandler(iostream)
logger.addHandler(lh)
lh.setLevel(logging.DEBUG)
lh.setFormatter(formatter)


random_coordinates = {
    "cartesian": tuples(
        floats(min_value=0, max_value=1),  # x
        floats(min_value=0, max_value=1),  # y
        floats(min_value=0, max_value=1),  # z
    ),
    "spherical": tuples(
        floats(min_value=0, max_value=1),  # r
        floats(min_value=0, max_value=180),  # theta
        floats(min_value=0, max_value=180),  # phi
    ),
    "cylindrical": tuples(
        floats(min_value=0, max_value=1),  # rho
        floats(min_value=0, max_value=180),  # phi
        floats(min_value=0, max_value=1),  # z
    ),
}


def test_instantiation_from_names(
    magnet_axes_instances, request: FixtureRequest
) -> None:
    """
    Instantiate AMI430_3D instrument from the three mock instruments
    representing current drivers for the x, y, and z directions by their
    names as opposed from their instances.
    """
    mag_x, mag_y, mag_z = magnet_axes_instances
    request.addfinalizer(AMIModel4303D.close_all)

    driver = AMIModel4303D("AMI430_3D", mag_x.name, mag_y.name, mag_z.name, field_limit)

    assert driver._instrument_x is mag_x
    assert driver._instrument_y is mag_y
    assert driver._instrument_z is mag_z


def test_instantiation_compat_classes(request: FixtureRequest) -> None:
    """
    Test that we can instantiate drivers using the old names
    """
    request.addfinalizer(AMIModel4303D.close_all)
    request.addfinalizer(AMI430_3D.close_all)
    mag_x = AMI430(
        "x", address="GPIB::1::INSTR", pyvisa_sim_file="AMI430.yaml", terminator="\n"
    )
    mag_y = AMI430(
        "y", address="GPIB::2::INSTR", pyvisa_sim_file="AMI430.yaml", terminator="\n"
    )
    mag_z = AMI430(
        "z", address="GPIB::3::INSTR", pyvisa_sim_file="AMI430.yaml", terminator="\n"
    )

    driver = AMI430_3D("AMI430_3D", mag_x.name, mag_y.name, mag_z.name, field_limit)

    assert driver._instrument_x is mag_x
    assert driver._instrument_y is mag_y
    assert driver._instrument_z is mag_z


def test_visa_interaction(request: FixtureRequest) -> None:
    """
    Test that closing one instrument we can still use the other simulated instruments.
    """
    request.addfinalizer(AMIModel4303D.close_all)
    mag_x = AMIModel430(
        "x", address="GPIB::1::INSTR", pyvisa_sim_file="AMI430.yaml", terminator="\n"
    )
    mag_y = AMIModel430(
        "y", address="GPIB::2::INSTR", pyvisa_sim_file="AMI430.yaml", terminator="\n"
    )
    mag_z = AMIModel430(
        "z", address="GPIB::3::INSTR", pyvisa_sim_file="AMI430.yaml", terminator="\n"
    )

    default_field_value = 0.123

    assert mag_x.field() == default_field_value
    assert mag_y.field() == default_field_value
    assert mag_z.field() == default_field_value

    mag_x.field(0.1)
    mag_y.field(0.2)
    mag_z.field(0.3)

    assert mag_x.field() == 0.1
    assert mag_y.field() == 0.2
    assert mag_z.field() == 0.3

    mag_x.close()
    # closing x should not change y or z
    assert mag_y.field() == 0.2
    assert mag_z.field() == 0.3


def test_sim_visa_reset_on_fully_closed(request: FixtureRequest) -> None:
    """
    Test that closing all instruments defined in a yaml file will reset the
    state of all the instruments.
    """
    request.addfinalizer(AMIModel4303D.close_all)
    mag_x = AMIModel430(
        "x", address="GPIB::1::INSTR", pyvisa_sim_file="AMI430.yaml", terminator="\n"
    )
    mag_y = AMIModel430(
        "y", address="GPIB::2::INSTR", pyvisa_sim_file="AMI430.yaml", terminator="\n"
    )
    mag_z = AMIModel430(
        "z", address="GPIB::3::INSTR", pyvisa_sim_file="AMI430.yaml", terminator="\n"
    )

    default_field_value = 0.123

    assert mag_x.field() == default_field_value
    assert mag_y.field() == default_field_value
    assert mag_z.field() == default_field_value

    mag_x.field(0.1)
    mag_y.field(0.2)
    mag_z.field(0.3)

    assert mag_x.field() == 0.1
    assert mag_y.field() == 0.2
    assert mag_z.field() == 0.3

    mag_x.close()
    mag_y.close()
    mag_z.close()

    # all are closed so instruments should be reset
    mag_x = AMIModel430(
        "x", address="GPIB::1::INSTR", pyvisa_sim_file="AMI430.yaml", terminator="\n"
    )
    mag_y = AMIModel430(
        "y", address="GPIB::2::INSTR", pyvisa_sim_file="AMI430.yaml", terminator="\n"
    )
    mag_z = AMIModel430(
        "z", address="GPIB::3::INSTR", pyvisa_sim_file="AMI430.yaml", terminator="\n"
    )

    assert mag_x.field() == default_field_value
    assert mag_y.field() == default_field_value
    assert mag_z.field() == default_field_value


def test_instantiation_from_name_of_nonexistent_ami_instrument(
    magnet_axes_instances, request: FixtureRequest
) -> None:
    mag_x, mag_y, mag_z = magnet_axes_instances
    request.addfinalizer(AMIModel4303D.close_all)

    non_existent_instrument = mag_y.name + "foo"

    with pytest.raises(
        KeyError, match=f"with name {non_existent_instrument} does not exist"
    ):
        AMIModel4303D(
            "AMI430_3D", mag_x.name, non_existent_instrument, mag_z.name, field_limit
        )


def test_instantiation_from_name_of_existing_non_ami_instrument(
    magnet_axes_instances, request: FixtureRequest
) -> None:
    mag_x, mag_y, mag_z = magnet_axes_instances
    request.addfinalizer(AMIModel4303D.close_all)

    non_ami_existing_instrument = Instrument("foo")

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"Instrument {non_ami_existing_instrument.name} is "
            f"{type(non_ami_existing_instrument)} but {AMIModel430} "
            f"was requested"
        ),
    ):
        AMIModel4303D(
            "AMI430_3D",
            mag_x.name,
            non_ami_existing_instrument.name,
            mag_z.name,
            field_limit,
        )


def test_instantiation_from_badly_typed_argument(
    magnet_axes_instances, request: FixtureRequest
) -> None:
    mag_x, mag_y, mag_z = magnet_axes_instances
    request.addfinalizer(AMIModel4303D.close_all)

    badly_typed_instrument_z_argument = 123

    with pytest.raises(ValueError, match="instrument_z argument is neither of those"):
        AMIModel4303D(
            "AMI430_3D",
            mag_x.name,
            mag_y,
            badly_typed_instrument_z_argument,  # type: ignore[arg-type]
            field_limit,
        )


@given(set_target=random_coordinates["cartesian"])
@settings(
    max_examples=10,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
    deadline=None,
)
def test_cartesian_sanity(current_driver, set_target) -> None:
    """
    A sanity check to see if the driver remember vectors in any random
    configuration in cartesian coordinates
    """
    current_driver.cartesian(set_target)
    get_target = current_driver.cartesian()

    assert np.allclose(set_target, get_target)

    # test that we can get the individual coordinates
    x = current_driver.x()
    y = current_driver.y()
    z = current_driver.z()
    assert np.allclose(set_target, [x, y, z])


@given(set_target=random_coordinates["spherical"])
@settings(
    max_examples=10,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
    deadline=None,
)
def test_spherical_sanity(current_driver, set_target) -> None:
    """
    A sanity check to see if the driver remember vectors in any random
    configuration in spherical coordinates
    """
    current_driver.spherical(set_target)
    get_target = current_driver.spherical()

    assert np.allclose(set_target, get_target)

    # test that we can get the individual coordinates
    r = current_driver.field()
    theta = current_driver.theta()
    phi = current_driver.phi()
    assert np.allclose(set_target, [r, theta, phi])


@given(set_target=random_coordinates["cylindrical"])
@settings(
    max_examples=10,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
    deadline=None,
)
def test_cylindrical_sanity(current_driver, set_target) -> None:
    """
    A sanity check to see if the driver remember vectors in any random
    configuration in cylindrical coordinates
    """
    current_driver.cylindrical(set_target)
    get_target = current_driver.cylindrical()

    assert np.allclose(set_target, get_target)

    # test that we can get the individual coordinates
    rho = current_driver.rho()
    z = current_driver.z()
    phi = current_driver.phi()
    assert np.allclose(set_target, [rho, phi, z])


# add some examples where floating point math results
# in z > r due to round off errors and ensure
# we handle them correctly
@example((0, 0, 3.729170476738041e-155))
@given(set_target=random_coordinates["cartesian"])
@settings(
    max_examples=10,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
    deadline=None,
)
def test_cartesian_setpoints(current_driver, set_target) -> None:
    """
    Check that the individual x, y, z instruments are getting the set
    points as intended. This test is very similar to the sanity test, but
    adds in the FieldVector as well.
    """
    current_driver.cartesian(set_target)

    x = current_driver.x()
    y = current_driver.y()
    z = current_driver.z()

    get_target = dict(zip(("x", "y", "z"), (x, y, z)))

    set_vector = FieldVector(*set_target)
    get_vector = FieldVector(**get_target)
    assert set_vector.is_equal(get_vector)


@given(set_target=random_coordinates["spherical"])
@settings(
    max_examples=10,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
    deadline=None,
)
def test_spherical_setpoints(current_driver, set_target) -> None:
    """
    Check that the individual x, y, z instruments are getting the set
    points as intended. This test is very similar to the sanity test, but
    adds in the FieldVector as well.
    """
    current_driver.spherical(set_target)

    r = current_driver.field()
    theta = current_driver.theta()
    phi = current_driver.phi()

    get_target = dict(zip(("r", "theta", "phi"), (r, theta, phi)))
    set_target = dict(zip(("r", "theta", "phi"), set_target))

    set_vector = FieldVector(**set_target)
    get_vector = FieldVector(**get_target)
    assert set_vector.is_equal(get_vector)

# add some examples where floating point math results
# in z > r due to round off errors and ensure
# we handle them correctly
@example((0, 0, 3.729170476738041e-155))
@given(set_target=random_coordinates["cylindrical"])
@settings(
    max_examples=10,
    deadline=500,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)
def test_cylindrical_setpoints(current_driver, set_target) -> None:
    """
    Check that the individual x, y, z instruments are getting the set
    points as intended. This test is very similar to the sanity test, but
    adds in the FieldVector as well.
    """
    current_driver.cylindrical(set_target)

    rho = current_driver.rho()
    z = current_driver.z()
    phi = current_driver.phi()

    get_target = dict(zip(("rho", "phi", "z"), (rho, phi, z)))
    set_target = dict(zip(("rho", "phi", "z"), set_target))

    set_vector = FieldVector(**set_target)
    get_vector = FieldVector(**get_target)
    assert set_vector.is_equal(get_vector)


@given(set_target=random_coordinates["cartesian"])
@settings(
    max_examples=10,
    deadline=500,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)
def test_measured(current_driver, set_target) -> None:
    """
    Simply call the measurement methods and verify that no exceptions
    are raised.
    """
    current_driver.cartesian(set_target)

    cartesian = current_driver.cartesian_measured()

    cartesian_x = current_driver.x_measured()
    cartesian_y = current_driver.y_measured()
    cartesian_z = current_driver.z_measured()

    assert np.allclose(cartesian, [cartesian_x, cartesian_y, cartesian_z])
    assert FieldVector(*set_target).is_equal(
        FieldVector(x=cartesian_x, y=cartesian_y, z=cartesian_z)
    )

    spherical = current_driver.spherical_measured()
    spherical_field = current_driver.field_measured()
    spherical_theta = current_driver.theta_measured()
    spherical_phi = current_driver.phi_measured()

    assert FieldVector(*set_target).is_equal(
        FieldVector(r=spherical_field, theta=spherical_theta, phi=spherical_phi)
    )
    assert np.allclose(spherical, [spherical_field, spherical_theta, spherical_phi])

    cylindrical = current_driver.cylindrical_measured()
    cylindrical_rho = current_driver.rho_measured()

    assert FieldVector(*set_target).is_equal(
        FieldVector(rho=cylindrical_rho, phi=spherical_phi, z=cartesian_z)
    )
    assert np.allclose(cylindrical, [cylindrical_rho, spherical_phi, cartesian_z])


def get_ramp_down_order(messages: list[str]) -> list[str]:
    order = []

    for msg in messages:
        if "CONF:FIELD:TARG" not in msg:
            continue

        g = re.search(r"\[(.*).*\] Writing: CONF:FIELD:TARG", msg)
        if g is None:
            raise RuntimeError(
                f"No match found in {msg!r} when getting ramp down order"
            )
        name = g.groups()[0]
        order.append(name)

    return order


def test_ramp_down_first(current_driver, caplog: LogCaptureFixture) -> None:
    """
    To prevent quenching of the magnets, we need the driver to always
    be within the field limits. Part of the strategy of making sure
    that this is the case includes always performing ramp downs
    first. For example, if magnets x and z need to be ramped up, while
    magnet y needs to be ramped down, ramp down magnet y first and
    then ramp up x and z.  This is tested here.
    """
    names = ["x", "y", "z"]
    set_point = np.array([0.3, 0.4, 0.5])
    # Put the magnets in a well defined state
    current_driver.cartesian(set_point)
    # We begin with ramping down x first while ramping up y and z
    delta = np.array([-0.1, 0.1, 0.1])

    with caplog.at_level(logging.DEBUG, logger=LOG_NAME):
        for count, ramp_down_name in enumerate(names):
            # The second iteration will ramp down y while ramping up x and z.
            set_point += np.roll(delta, count)
            # Check if y is adjusted first.
            # We will perform the same test with z in the third iteration

            # clear the message stream
            caplog.clear()

            # make a ramp
            current_driver.cartesian(set_point)
            # get the logging outputs from the instruments.
            messages = [record.message for record in caplog.records]
            # get the order in which the ramps down occur
            order = get_ramp_down_order(messages)
            # the first one should be the one for which delta < 0
            assert order[0][0] == names[count]


def test_field_limit_exception(current_driver) -> None:
    """
    Test that an exception is raised if we intentionally set the field
    beyond the limits. Together with the no_test_ramp_down_first test
    this should prevent us from ever exceeding set point limits.  In
    this test we generate a regular grid in three-D and assert that
    the driver can be set to a set point if any of of the requirements
    given by field_limit is satisfied. An error *must* be raised if
    none of the safety limits are satisfied.
    """
    x = np.linspace(-3, 3, 11)
    y = np.copy(x)
    z = np.copy(x)
    set_points = zip(*(i.flatten() for i in np.meshgrid(x, y, z)))

    for set_point in set_points:
        should_not_raise = any([is_safe(*set_point) for is_safe in field_limit])

        if should_not_raise:
            current_driver.cartesian(set_point)
        else:
            with pytest.raises(Exception) as excinfo:
                current_driver.cartesian(set_point)

            assert "field would exceed limit" in excinfo.value.args[0]
            vals_and_setpoints = zip(current_driver.cartesian(), set_point)
            belief = not (all([val == sp for val, sp in vals_and_setpoints]))
            assert belief


def test_cylindrical_poles(current_driver) -> None:
    """
    Test that the phi coordinate is remembered even if the resulting
    vector is equivalent to the null vector
    """
    rho, phi, z = 0.4, 30.0, 0.5
    # This is equivalent to the null vector
    current_driver.cylindrical((0.0, phi, 0.0))
    current_driver.rho(rho)
    current_driver.z(z)

    # after setting the rho and z values we should have the vector as
    # originally intended.
    rho_m, phi_m, z_m = current_driver.cylindrical()

    assert np.allclose([rho_m, phi_m, z_m], [rho, phi, z])


def test_spherical_poles(current_driver) -> None:
    """
    Test that the theta and phi coordinates are remembered even if the
    resulting vector is equivalent to the null vector
    """

    field, theta, phi = 0.5, 30.0, 50.0
    # If field=0, then this is equivalent to the null vector
    current_driver.spherical((0.0, theta, phi))
    # x, y, z = (0, 0, 0)
    current_driver.field(field)

    # after setting the field we should have the vector as
    # originally intended.
    field_m, theta_m, phi_m = current_driver.spherical()
    assert np.allclose([field_m, theta_m, phi_m], [field, theta, phi])


def test_ramp_rate_exception(current_driver) -> None:
    """
    Test that an exception is raised if we try to set the ramp rate
    to a higher value than is allowed
    """
    ix = current_driver._instrument_x
    max_ramp_rate = ix.field_ramp_limit()
    target_ramp_rate = max_ramp_rate + 0.01

    with pytest.raises(ValueError, match="is above the ramp rate limit of"):
        ix.ramp_rate(target_ramp_rate)


def test_simultaneous_ramp_mode_does_not_reset_individual_axis_ramp_rates_if_nonblocking_ramp(
    current_driver, caplog: LogCaptureFixture, request: FixtureRequest
) -> None:
    ami3d = current_driver

    ami3d.cartesian((0.0, 0.0, 0.0))

    restore_parameters_stack = ExitStack()
    request.addfinalizer(restore_parameters_stack.close)

    restore_parameters_stack.callback(ami3d.cartesian, (0.0, 0.0, 0.0))

    restore_parameters_stack.enter_context(
        ami3d._instrument_x.ramp_rate.restore_at_exit()
    )
    restore_parameters_stack.enter_context(
        ami3d._instrument_y.ramp_rate.restore_at_exit()
    )
    restore_parameters_stack.enter_context(
        ami3d._instrument_z.ramp_rate.restore_at_exit()
    )

    restore_parameters_stack.enter_context(ami3d.ramp_mode.set_to("simultaneous"))

    restore_parameters_stack.enter_context(ami3d.block_during_ramp.set_to(False))

    # Set individual ramp rates to known values
    ami3d._instrument_x.ramp_rate(0.09)
    ami3d._instrument_y.ramp_rate(0.10)
    ami3d._instrument_z.ramp_rate(0.11)

    ami3d.vector_ramp_rate(0.05)

    with caplog.at_level(logging.DEBUG, logger=LOG_NAME):

        # Initiate the simultaneous ramp
        ami3d.cartesian((0.5, 0.5, 0.5))

        # Assert the individual axes ramp rates were changed and not reverted
        # to the known values set earlier
        assert ami3d._instrument_x.ramp_rate() != 0.09
        assert ami3d._instrument_y.ramp_rate() != 0.10
        assert ami3d._instrument_z.ramp_rate() != 0.11

        # Assert the expected values of the ramp rates of the individual axes
        # set by the simultaneous ramp based on the vector_ramp_rate and the
        # setpoint magnetic field
        expected_ramp_rate = pytest.approx(
            0.5 / np.linalg.norm(ami3d.cartesian(), ord=2) * ami3d.vector_ramp_rate()
        )
        assert ami3d._instrument_x.ramp_rate() == expected_ramp_rate
        assert ami3d._instrument_y.ramp_rate() == expected_ramp_rate
        assert ami3d._instrument_z.ramp_rate() == expected_ramp_rate

    messages = [record.message for record in caplog.records]

    expected_log_fragment = "Simultaneous ramp: not blocking until ramp is finished"
    messages_with_expected_fragment = tuple(
        message for message in messages if expected_log_fragment in message
    )
    assert (
        len(messages_with_expected_fragment) == 1
    ), f"found: {messages_with_expected_fragment}"

    unexpected_log_fragment = "Restoring individual axes ramp rates"
    messages_with_unexpected_fragment = tuple(
        message for message in messages if unexpected_log_fragment in message
    )
    assert (
        len(messages_with_unexpected_fragment) == 0
    ), f"found: {messages_with_unexpected_fragment}"

    # However, calling ``wait_while_all_axes_ramping`` DOES restore the
    # individual ramp rates

    with caplog.at_level(logging.DEBUG, logger=LOG_NAME):
        ami3d.wait_while_all_axes_ramping()

    messages_2 = [record.message for record in caplog.records]

    expected_log_fragment_2 = "Restoring individual axes ramp rates"
    messages_with_expected_fragment_2 = tuple(
        message for message in messages_2 if expected_log_fragment_2 in message
    )
    assert (
        len(messages_with_expected_fragment_2) == 1
    ), f"found: {messages_with_expected_fragment_2}"

    # Assert calling ``wait_while_all_axes_ramping`` is possible

    ami3d.wait_while_all_axes_ramping()


def test_simultaneous_ramp_mode_resets_individual_axis_ramp_rates_if_blocking_ramp(
    current_driver, caplog: LogCaptureFixture, request: FixtureRequest
) -> None:
    ami3d = current_driver

    ami3d.cartesian((0.0, 0.0, 0.0))

    restore_parameters_stack = ExitStack()
    request.addfinalizer(restore_parameters_stack.close)

    restore_parameters_stack.callback(ami3d.cartesian, (0.0, 0.0, 0.0))

    restore_parameters_stack.enter_context(
        ami3d._instrument_x.ramp_rate.restore_at_exit()
    )
    restore_parameters_stack.enter_context(
        ami3d._instrument_y.ramp_rate.restore_at_exit()
    )
    restore_parameters_stack.enter_context(
        ami3d._instrument_z.ramp_rate.restore_at_exit()
    )

    restore_parameters_stack.enter_context(ami3d.ramp_mode.set_to("simultaneous"))

    restore_parameters_stack.enter_context(ami3d.block_during_ramp.set_to(True))

    with caplog.at_level(logging.DEBUG, logger=LOG_NAME):

        # Set individual ramp rates to known values
        ami3d._instrument_x.ramp_rate(0.09)
        ami3d._instrument_y.ramp_rate(0.10)
        ami3d._instrument_z.ramp_rate(0.11)

        ami3d.vector_ramp_rate(0.05)

        # Initiate the simultaneous ramp
        ami3d.cartesian((0.5, 0.5, 0.5))

        # Assert the individual axes ramp rates were reverted
        # to the known values set earlier
        assert ami3d._instrument_x.ramp_rate() == 0.09
        assert ami3d._instrument_y.ramp_rate() == 0.10
        assert ami3d._instrument_z.ramp_rate() == 0.11

    messages = [record.message for record in caplog.records]

    expected_log_fragment = "Restoring individual axes ramp rates"
    messages_with_expected_fragment = tuple(
        message for message in messages if expected_log_fragment in message
    )
    assert (
        len(messages_with_expected_fragment) == 1
    ), f"found: {messages_with_expected_fragment}"

    expected_log_fragment_2 = "Simultaneous ramp: blocking until ramp is finished"
    messages_with_expected_fragment_2 = tuple(
        message for message in messages if expected_log_fragment_2 in message
    )
    assert (
        len(messages_with_expected_fragment_2) == 1
    ), f"found: {messages_with_expected_fragment_2}"

    unexpected_log_fragment = "Simultaneous ramp: not blocking until ramp is finished"
    messages_with_unexpected_fragment = tuple(
        message for message in messages if unexpected_log_fragment in message
    )
    assert (
        len(messages_with_unexpected_fragment) == 0
    ), f"found: {messages_with_unexpected_fragment}"


def test_reducing_field_ramp_limit_reduces_a_higher_ramp_rate(ami430) -> None:
    """
    When reducing field_ramp_limit, the actual ramp_rate should also be
    reduced if the new field_ramp_limit is lower than the actual ramp_rate
    now.
    """
    factor = 0.8

    # The following fact is expected for the test
    assert ami430.ramp_rate() <= ami430.field_ramp_limit()

    # Set ramp_rate_limit to value that is smaller than the ramp_rate of now
    new_field_ramp_limit = ami430.ramp_rate() * factor
    ami430.field_ramp_limit(new_field_ramp_limit)

    # Assert that the ramp_rate changed to fit within the new field_ramp_limit
    assert ami430.ramp_rate() <= ami430.field_ramp_limit()

    # Well, actually, the new ramp_rate is equal to the new field_ramp_limit
    assert ami430.ramp_rate() == ami430.field_ramp_limit()


def test_reducing_current_ramp_limit_reduces_a_higher_ramp_rate(ami430) -> None:
    """
    When reducing current_ramp_limit, the actual ramp_rate should also be
    reduced if the new current_ramp_limit is lower than the actual ramp_rate
    now (with respect to field/current conversion).
    """
    factor = 0.8

    # The following fact is expected for the test
    assert ami430.ramp_rate() <= ami430.current_ramp_limit() * ami430.coil_constant()

    # Set ramp_rate_limit to value that is smaller than the ramp_rate of now
    new_current_ramp_limit = ami430.ramp_rate() * factor / ami430.coil_constant()
    ami430.current_ramp_limit(new_current_ramp_limit)

    # Assert that the ramp_rate changed to fit within the new field_ramp_limit
    assert ami430.ramp_rate() <= ami430.field_ramp_limit()
    assert ami430.ramp_rate() <= ami430.current_ramp_limit() * ami430.coil_constant()

    # Well, actually, the new ramp_rate is equal to the new field_ramp_limit
    assert ami430.ramp_rate() == ami430.field_ramp_limit()


def test_reducing_field_ramp_limit_keeps_a_lower_ramp_rate_as_is(ami430) -> None:
    """
    When reducing field_ramp_limit, the actual ramp_rate should remain
    if the new field_ramp_limit is higher than the actual ramp_rate now.
    """
    factor = 1.2

    # The following fact is expected for the test
    assert ami430.ramp_rate() <= ami430.field_ramp_limit()

    old_ramp_rate = ami430.ramp_rate()

    # Set ramp_rate_limit to value that is larger than the ramp_rate of now
    new_field_ramp_limit = ami430.ramp_rate() * factor
    ami430.field_ramp_limit(new_field_ramp_limit)

    # Assert that the ramp_rate remained within the new field_ramp_limit
    assert ami430.ramp_rate() <= ami430.field_ramp_limit()

    # Assert that ramp_rate hasn't actually changed
    assert ami430.ramp_rate() == old_ramp_rate


def test_reducing_current_ramp_limit_keeps_a_lower_ramp_rate_as_is(ami430) -> None:
    """
    When reducing current_ramp_limit, the actual ramp_rate should remain
    if the new current_ramp_limit is higher than the actual ramp_rate now
    (with respect to field/current conversion).
    """
    factor = 1.2

    # The following fact is expected for the test
    assert ami430.ramp_rate() <= ami430.current_ramp_limit() * ami430.coil_constant()

    old_ramp_rate = ami430.ramp_rate()

    # Set ramp_rate_limit to value that is larger than the ramp_rate of now
    new_current_ramp_limit = ami430.ramp_rate() * factor / ami430.coil_constant()
    ami430.current_ramp_limit(new_current_ramp_limit)

    # Assert that the ramp_rate remained within the new field_ramp_limit
    assert ami430.ramp_rate() <= ami430.field_ramp_limit()
    assert ami430.ramp_rate() <= ami430.current_ramp_limit() * ami430.coil_constant()

    # Assert that ramp_rate hasn't actually changed
    assert ami430.ramp_rate() == old_ramp_rate


def test_blocking_ramp_parameter(current_driver, caplog: LogCaptureFixture) -> None:

    assert current_driver.block_during_ramp() is True

    with caplog.at_level(logging.DEBUG, logger=LOG_NAME):
        current_driver.cartesian((0, 0, 0))
        caplog.clear()
        current_driver.cartesian((0, 0, 1))

        messages = [record.message for record in caplog.records]
        assert messages[-1] == "[z(AMIModel430)] Finished blocking ramp"
        assert messages[-6] == "[z(AMIModel430)] Starting blocking ramp of z to 1.0"

        caplog.clear()
        current_driver.block_during_ramp(False)
        current_driver.cartesian((0, 0, 0))
        messages = [record.message for record in caplog.records]

        assert len([mssg for mssg in messages if "blocking" in mssg]) == 0


def test_current_and_field_params_interlink_at_init(ami430) -> None:
    """
    Test that the values of the ``coil_constant``-dependent parameters
    are correctly proportional to each other at the initialization of the
    instrument driver.
    """
    coil_constant = ami430.coil_constant()
    current_ramp_limit = ami430.current_ramp_limit()
    field_ramp_limit = ami430.field_ramp_limit()
    current_limit = ami430.current_limit()
    field_limit = ami430.field_limit()

    np.testing.assert_almost_equal(field_ramp_limit, current_ramp_limit * coil_constant)

    np.testing.assert_almost_equal(field_limit, current_limit * coil_constant)


def test_current_and_field_params_interlink__change_current_ramp_limit(
    ami430, factor=0.9
) -> None:
    """
    Test that after changing ``current_ramp_limit``, the values of the
    ``field_*`` parameters change proportionally, ``coil__constant`` remains
    the same. At the end just ensure that the values of the
    ``coil_constant``-dependent parameters are correctly proportional to each
    other.
    """
    coil_constant_old = ami430.coil_constant()
    current_ramp_limit_old = ami430.current_ramp_limit()
    field_ramp_limit_old = ami430.field_ramp_limit()
    current_limit_old = ami430.current_limit()
    field_limit_old = ami430.field_limit()

    current_ramp_limit_new = current_ramp_limit_old * factor

    ami430.current_ramp_limit(current_ramp_limit_new)

    field_ramp_limit_new_expected = field_ramp_limit_old * factor

    current_ramp_limit = ami430.current_ramp_limit()
    field_ramp_limit = ami430.field_ramp_limit()
    coil_constant = ami430.coil_constant()
    current_limit = ami430.current_limit()
    field_limit = ami430.field_limit()

    # The following parameters are expected to change
    np.testing.assert_almost_equal(current_ramp_limit, current_ramp_limit_new)
    np.testing.assert_almost_equal(field_ramp_limit, field_ramp_limit_new_expected)

    # The following parameters are not expected to change
    np.testing.assert_almost_equal(coil_constant, coil_constant_old)
    np.testing.assert_almost_equal(current_limit, current_limit_old)
    np.testing.assert_almost_equal(field_limit, field_limit_old)

    # Proportions are expected to hold between field and current parameters
    np.testing.assert_almost_equal(field_ramp_limit, current_ramp_limit * coil_constant)
    np.testing.assert_almost_equal(field_limit, current_limit * coil_constant)


def test_current_and_field_params_interlink__change_field_ramp_limit(
    ami430, factor=0.9
) -> None:
    """
    Test that after changing ``field_ramp_limit``, the values of the
    ``current_*`` parameters change proportionally, ``coil__constant`` remains
    the same. At the end just ensure that the values of the
    ``coil_constant``-dependent parameters are correctly proportional to each
    other.
    """
    coil_constant_old = ami430.coil_constant()
    current_ramp_limit_old = ami430.current_ramp_limit()
    field_ramp_limit_old = ami430.field_ramp_limit()
    current_limit_old = ami430.current_limit()
    field_limit_old = ami430.field_limit()

    field_ramp_limit_new = field_ramp_limit_old * factor

    ami430.field_ramp_limit(field_ramp_limit_new)

    current_ramp_limit_new_expected = current_ramp_limit_old * factor

    current_ramp_limit = ami430.current_ramp_limit()
    field_ramp_limit = ami430.field_ramp_limit()
    coil_constant = ami430.coil_constant()
    current_limit = ami430.current_limit()
    field_limit = ami430.field_limit()

    # The following parameters are expected to change
    np.testing.assert_almost_equal(field_ramp_limit, field_ramp_limit_new)
    np.testing.assert_almost_equal(current_ramp_limit, current_ramp_limit_new_expected)

    # The following parameters are not expected to change
    np.testing.assert_almost_equal(coil_constant, coil_constant_old)
    np.testing.assert_almost_equal(current_limit, current_limit_old)
    np.testing.assert_almost_equal(field_limit, field_limit_old)

    # Proportions are expected to hold between field and current parameters
    np.testing.assert_almost_equal(field_ramp_limit, current_ramp_limit * coil_constant)
    np.testing.assert_almost_equal(field_limit, current_limit * coil_constant)


def test_current_and_field_params_interlink__change_coil_constant(
    ami430, factor: float = 3
) -> None:
    """
    Test that after changing ``change_coil_constant``, the values of the
    ``current_*`` parameters remain the same while the values of the
    ``field_*`` parameters change proportionally. At the end just ensure that
    the values of the ``coil_constant``-dependent parameters are correctly
    proportional to each other.
    """
    coil_constant_old = ami430.coil_constant()
    current_ramp_limit_old = ami430.current_ramp_limit()
    field_ramp_limit_old = ami430.field_ramp_limit()
    current_limit_old = ami430.current_limit()
    field_limit_old = ami430.field_limit()

    coil_constant_new = coil_constant_old * factor

    ami430.coil_constant(coil_constant_new)

    current_ramp_limit_new_expected = current_ramp_limit_old
    current_limit_new_expected = current_limit_old
    field_ramp_limit_new_expected = field_ramp_limit_old * factor
    field_limit_new_expected = field_limit_old * factor

    current_ramp_limit = ami430.current_ramp_limit()
    field_ramp_limit = ami430.field_ramp_limit()
    coil_constant = ami430.coil_constant()
    current_limit = ami430.current_limit()
    field_limit = ami430.field_limit()

    # The following parameters are expected to change
    np.testing.assert_almost_equal(coil_constant, coil_constant_new)
    np.testing.assert_almost_equal(current_ramp_limit, current_ramp_limit_new_expected)
    np.testing.assert_almost_equal(field_ramp_limit, field_ramp_limit_new_expected)
    np.testing.assert_almost_equal(current_limit, current_limit_new_expected)
    np.testing.assert_almost_equal(field_limit, field_limit_new_expected)

    # Proportions are expected to hold between field and current parameters
    np.testing.assert_almost_equal(field_ramp_limit, current_ramp_limit * coil_constant)
    np.testing.assert_almost_equal(field_limit, current_limit * coil_constant)


def test_current_and_field_params_interlink__permutations_of_tests(ami430) -> None:
    """
    As per one of the user's request, the
    test_current_and_field_params_interlink__* tests are executed here with
    arbitrary 'factor's and with all permutations. This test ensures the
    robustness of the driver even more.

    Note that the 'factor's are randomized "manually" because of the
    possibility to hit the limits of the parameters.
    """
    with warnings.catch_warnings():
        # this is to avoid AMI430Warning about "maximum ramp rate", which
        # may show up but is not relevant to this test
        warnings.simplefilter("ignore", category=AMI430Warning)

        test_current_and_field_params_interlink_at_init(ami430)

        test_current_and_field_params_interlink__change_coil_constant(
            ami430, factor=1.2
        )
        test_current_and_field_params_interlink__change_field_ramp_limit(
            ami430, factor=1.0023
        )
        test_current_and_field_params_interlink__change_current_ramp_limit(
            ami430, factor=0.98
        )

        test_current_and_field_params_interlink__change_coil_constant(
            ami430, factor=1.53
        )
        test_current_and_field_params_interlink__change_current_ramp_limit(
            ami430, factor=2.0
        )
        test_current_and_field_params_interlink__change_field_ramp_limit(
            ami430, factor=0.633
        )

        test_current_and_field_params_interlink__change_field_ramp_limit(
            ami430, factor=1.753
        )
        test_current_and_field_params_interlink__change_coil_constant(
            ami430, factor=0.876
        )
        test_current_and_field_params_interlink__change_current_ramp_limit(
            ami430, factor=4.6
        )

        test_current_and_field_params_interlink__change_coil_constant(
            ami430, factor=1.87
        )
        test_current_and_field_params_interlink__change_current_ramp_limit(
            ami430, factor=2.11
        )
        test_current_and_field_params_interlink__change_field_ramp_limit(
            ami430, factor=1.0020
        )

        test_current_and_field_params_interlink__change_field_ramp_limit(
            ami430, factor=0.42
        )
        test_current_and_field_params_interlink__change_current_ramp_limit(
            ami430, factor=3.1415
        )
        test_current_and_field_params_interlink__change_coil_constant(
            ami430, factor=1.544
        )

        test_current_and_field_params_interlink__change_coil_constant(
            ami430, factor=0.12
        )
        test_current_and_field_params_interlink__change_field_ramp_limit(
            ami430, factor=0.4422
        )
        test_current_and_field_params_interlink__change_current_ramp_limit(
            ami430, factor=0.00111
        )


class PDict(TypedDict):
    argvalues: list[Any]
    ids: list[str]


def _parametrization_kwargs() -> PDict:
    kwargs: PDict = {"argvalues": [], "ids": []}

    for type_constructor, type_name in zip(
        (
            (int, float)
            + numpy_concrete_ints
            + numpy_non_concrete_ints_instantiable
            + numpy_concrete_floats
        ),
        (
            ["int", "float"]
            + [str(t) for t in numpy_concrete_ints]
            + [str(t) for t in numpy_non_concrete_ints_instantiable]
            + [str(t) for t in numpy_concrete_floats]
        ),
    ):
        kwargs["argvalues"].append(type_constructor(2.2))
        kwargs["ids"].append(type_name)

    return kwargs


@pytest.mark.parametrize("field_limit", **_parametrization_kwargs())
def test_numeric_field_limit(
    magnet_axes_instances, field_limit, request: FixtureRequest
) -> None:
    mag_x, mag_y, mag_z = magnet_axes_instances
    ami = AMIModel4303D("AMI430_3D", mag_x, mag_y, mag_z, field_limit)
    request.addfinalizer(ami.close)

    assert isinstance(ami._field_limit, float)

    target_within_limit = (field_limit * 0.95, 0, 0)
    ami.cartesian(target_within_limit)

    target_outside_limit = (field_limit * 1.05, 0, 0)
    with pytest.raises(
        ValueError, match="_set_fields aborted; field would exceed limit"
    ):
        ami.cartesian(target_outside_limit)


def test_ramp_rate_units_and_field_units_at_init(ami430) -> None:
    """
    Test values of ramp_rate_units and field_units parameters at init,
    and the units of other parameters which depend on the
    values of ramp_rate_units and field_units parameters.
    """
    initial_ramp_rate_units = ami430.ramp_rate_units()
    initial_field_units = ami430.field_units()

    assert initial_ramp_rate_units == "seconds"
    assert initial_field_units == "tesla"

    assert ami430.coil_constant.unit == "T/A"
    assert ami430.field_limit.unit == "T"
    assert ami430.field.unit == "T"
    assert ami430.setpoint.unit == "T"
    assert ami430.ramp_rate.unit == "T/s"
    assert ami430.current_ramp_limit.unit == "A/s"
    assert ami430.field_ramp_limit.unit == "T/s"


@pytest.mark.parametrize(
    ("new_value", "unit_string", "scale"),
    (("seconds", "s", 1), ("minutes", "min", 1 / 60)),
    ids=("seconds", "minutes"),
)
def test_change_ramp_rate_units_parameter(
    ami430, new_value, unit_string, scale
) -> None:
    """
    Test that changing value of ramp_rate_units parameter is reflected in
    settings of other magnet parameters.
    """
    coil_constant_unit = ami430.coil_constant.unit
    field_limit_unit = ami430.field_limit.unit
    field_unit = ami430.field.unit
    setpoint_unit = ami430.setpoint.unit
    coil_constant_timestamp = ami430.coil_constant.get_latest.get_timestamp()
    # this prevents possible flakiness of the timestamp comparison
    # later in the test that may originate from the not-enough resolution
    # of the time function used in `Parameter` and `GetLatest` classes
    time.sleep(2 * _time_resolution)

    ami430.ramp_rate_units(new_value)

    ramp_rate_units__actual = ami430.ramp_rate_units()
    assert ramp_rate_units__actual == new_value

    assert ami430.coil_constant.unit == coil_constant_unit
    assert ami430.field_limit.unit == field_limit_unit
    assert ami430.field.unit == field_unit
    assert ami430.setpoint.unit == setpoint_unit

    assert ami430.ramp_rate.unit.endswith("/" + unit_string)
    assert ami430.current_ramp_limit.unit.endswith("/" + unit_string)
    assert ami430.field_ramp_limit.unit.endswith("/" + unit_string)

    assert ami430.current_ramp_limit.scale == scale

    # Assert `coil_constant` value has been updated
    assert ami430.coil_constant.get_latest.get_timestamp() > coil_constant_timestamp

    ami430.ramp_rate_units("seconds")


@pytest.mark.parametrize(
    ("new_value", "unit_string"),
    (("tesla", "T"), ("kilogauss", "kG")),
    ids=("tesla", "kilogauss"),
)
def test_change_field_units_parameter(ami430, new_value, unit_string) -> None:
    """
    Test that changing value of field_units parameter is reflected in
    settings of other magnet parameters.
    """
    current_ramp_limit_unit = ami430.current_ramp_limit.unit
    current_ramp_limit_scale = ami430.current_ramp_limit.scale
    coil_constant_timestamp = ami430.coil_constant.get_latest.get_timestamp()
    # this prevents possible flakiness of the timestamp comparison
    # later in the test that may originate from the not-enough resolution
    # of the time function used in `Parameter` and `GetLatest` classes
    time.sleep(2 * _time_resolution)

    ami430.field_units(new_value)

    field_units__actual = ami430.field_units()
    assert field_units__actual == new_value

    assert ami430.current_ramp_limit.unit == current_ramp_limit_unit
    assert ami430.current_ramp_limit.scale == current_ramp_limit_scale

    assert ami430.field_limit.unit == unit_string
    assert ami430.field.unit == unit_string
    assert ami430.setpoint.unit == unit_string

    assert ami430.coil_constant.unit.startswith(unit_string + "/")
    assert ami430.ramp_rate.unit.startswith(unit_string + "/")
    assert ami430.field_ramp_limit.unit.startswith(unit_string + "/")

    # Assert `coil_constant` value has been updated
    assert ami430.coil_constant.get_latest.get_timestamp() > coil_constant_timestamp

    ami430.field_units("tesla")


def test_switch_heater_enabled(ami430) -> None:
    assert ami430.switch_heater.enabled() is False
    ami430.switch_heater.enabled(True)
    assert ami430.switch_heater.enabled() is True
    ami430.switch_heater.enabled(False)
    assert ami430.switch_heater.enabled() is False
