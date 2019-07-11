import io
import numpy as np
import re
import pytest
from hypothesis import given, settings
from hypothesis.strategies import floats
from hypothesis.strategies import tuples
import logging
import warnings
from typing import List

import qcodes.instrument.sims as sims
from qcodes.instrument_drivers.american_magnetics.AMI430 import AMI430_3D, \
    AMI430Warning
from qcodes.instrument.ip_to_visa import AMI430_VISA
from qcodes.math.field_vector import FieldVector
from qcodes.utils.types import numpy_concrete_ints, numpy_concrete_floats, \
    numpy_non_concrete_ints_instantiable, \
    numpy_non_concrete_floats_instantiable


# If any of the field limit functions are satisfied we are in the safe zone.
# We can have higher field along the z-axis if x and y are zero.
field_limit = [
    lambda x, y, z: x == 0 and y == 0 and z < 3,
    lambda x, y, z: np.linalg.norm([x, y, z]) < 2
]

# path to the .yaml file containing the simulated instrument
visalib = sims.__file__.replace('__init__.py', 'AMI430.yaml@sim')


@pytest.fixture(scope='function')
def magnet_axes_instances():
    """
    Start three mock instruments representing current drivers for the x, y,
    and z directions.
    """
    mag_x = AMI430_VISA('x', address='GPIB::1::INSTR', visalib=visalib,
                        terminator='\n', port=1)
    mag_y = AMI430_VISA('y', address='GPIB::2::INSTR', visalib=visalib,
                        terminator='\n', port=1)
    mag_z = AMI430_VISA('z', address='GPIB::3::INSTR', visalib=visalib,
                        terminator='\n', port=1)

    yield mag_x, mag_y, mag_z

    mag_x.close()
    mag_y.close()
    mag_z.close()


@pytest.fixture(scope='function')
def current_driver(magnet_axes_instances):
    """
    Instantiate AMI430_3D instrument with the three mock instruments
    representing current drivers for the x, y, and z directions.
    """
    mag_x, mag_y, mag_z = magnet_axes_instances

    driver = AMI430_3D("AMI430-3D", mag_x, mag_y, mag_z, field_limit)

    yield driver

    driver.close()


@pytest.fixture(scope='function',
                params=(True, False))
def ami430(request):
    mag = AMI430_VISA('ami430', address='GPIB::1::INSTR', visalib=visalib,
                      terminator='\n', port=1,
                      has_current_rating=request.param)
    yield mag
    mag.close()


# here the original test has a homemade log system that we don't want to
# reproduce / write tests for. Instead, we use normal logging from our
# instrument.visa module
iostream = io.StringIO()
logger = logging.getLogger('qcodes.instrument.visa')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(created)s - %(message)s')
lh = logging.StreamHandler(iostream)
logger.addHandler(lh)
lh.setLevel(logging.DEBUG)
lh.setFormatter(formatter)


random_coordinates = {
    "cartesian": tuples(
        floats(min_value=0, max_value=1),  # x
        floats(min_value=0, max_value=1),  # y
        floats(min_value=0, max_value=1)  # z
    ),
    "spherical": tuples(
        floats(min_value=0, max_value=1),  # r
        floats(min_value=0, max_value=180),  # theta
        floats(min_value=0, max_value=180)  # phi
    ),
    "cylindrical": tuples(
        floats(min_value=0, max_value=1),  # rho
        floats(min_value=0, max_value=180),  # phi
        floats(min_value=0, max_value=1)  # z
    )
}


@given(set_target=random_coordinates["cartesian"])
@settings(max_examples=10)
def test_cartesian_sanity(current_driver, set_target):
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
@settings(max_examples=10)
def test_spherical_sanity(current_driver, set_target):
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
@settings(max_examples=10)
def test_cylindrical_sanity(current_driver, set_target):
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


@given(set_target=random_coordinates["cartesian"])
@settings(max_examples=10)
def test_cartesian_setpoints(current_driver, set_target):
    """
    Check that the individual x, y, z instruments are getting the set
    points as intended. This test is very similar to the sanity test, but
    adds in the FieldVector as well.
    """
    current_driver.cartesian(set_target)

    x = current_driver.x()
    y = current_driver.y()
    z = current_driver.z()

    get_target = dict(zip(('x', 'y', 'z'), (x, y, z)))

    set_vector = FieldVector(*set_target)
    get_vector = FieldVector(**get_target)
    assert set_vector.is_equal(get_vector)


@given(set_target=random_coordinates["spherical"])
@settings(max_examples=10)
def test_spherical_setpoints(current_driver, set_target):
    """
    Check that the individual x, y, z instruments are getting the set
    points as intended. This test is very similar to the sanity test, but
    adds in the FieldVector as well.
    """
    current_driver.spherical(set_target)

    r = current_driver.field()
    theta = current_driver.theta()
    phi = current_driver.phi()

    get_target = dict(zip(('r', 'theta', 'phi'), (r, theta, phi)))
    set_target = dict(zip(('r', 'theta', 'phi'), set_target))

    set_vector = FieldVector(**set_target)
    get_vector = FieldVector(**get_target)
    assert set_vector.is_equal(get_vector)


@given(set_target=random_coordinates["cylindrical"])
@settings(max_examples=10, deadline=500)
def test_cylindrical_setpoints(current_driver, set_target):
    """
    Check that the individual x, y, z instruments are getting the set
    points as intended. This test is very similar to the sanity test, but
    adds in the FieldVector as well.
    """
    current_driver.cylindrical(set_target)

    rho = current_driver.rho()
    z = current_driver.z()
    phi = current_driver.phi()

    get_target = dict(zip(('rho', 'phi', 'z'), (rho, phi, z)))
    set_target = dict(zip(('rho', 'phi', 'z'), set_target))

    set_vector = FieldVector(**set_target)
    get_vector = FieldVector(**get_target)
    assert set_vector.is_equal(get_vector)


@given(set_target=random_coordinates["cartesian"])
@settings(max_examples=10, deadline=500)
def test_measured(current_driver, set_target):
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
    assert FieldVector(*set_target).is_equal(FieldVector(x=cartesian_x,
                                                         y=cartesian_y,
                                                         z=cartesian_z))

    spherical = current_driver.spherical_measured()
    spherical_field = current_driver.field_measured()
    spherical_theta = current_driver.theta_measured()
    spherical_phi = current_driver.phi_measured()

    assert FieldVector(*set_target).is_equal(FieldVector(
        r=spherical_field,
        theta=spherical_theta,
        phi=spherical_phi)
    )
    assert np.allclose(spherical,
                       [spherical_field, spherical_theta, spherical_phi])

    cylindrical = current_driver.cylindrical_measured()
    cylindrical_rho = current_driver.rho_measured()

    assert FieldVector(*set_target).is_equal(FieldVector(rho=cylindrical_rho,
                                                         phi=spherical_phi,
                                                         z=cartesian_z))
    assert np.allclose(cylindrical, [cylindrical_rho,
                                     spherical_phi,
                                     cartesian_z])


def get_ramp_down_order(messages: List[str]) -> List[str]:
    order = []

    for msg in messages:
        if "CONF:FIELD:TARG" not in msg:
            continue

        g = re.search(r"\[(.*).*\] Writing: CONF:FIELD:TARG", msg)
        if g is None:
            raise RuntimeError(
                f"No match found in {msg!r} when getting ramp down order")
        name = g.groups()[0]
        order.append(name)

    return order


def test_ramp_down_first(current_driver, caplog):
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

    log_name = 'qcodes.instrument.base'

    with caplog.at_level(logging.DEBUG, logger=log_name):
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


def test_field_limit_exception(current_driver):
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
    set_points = zip(*[i.flatten() for i in np.meshgrid(x, y, z)])

    for set_point in set_points:
        should_not_raise = any([is_safe(*set_point)
                                for is_safe in field_limit])

        if should_not_raise:
            current_driver.cartesian(set_point)
        else:
            with pytest.raises(Exception) as excinfo:
                current_driver.cartesian(set_point)

            assert "field would exceed limit" in excinfo.value.args[0]
            vals_and_setpoints = zip(current_driver.cartesian(), set_point)
            belief = not(all([val == sp for val, sp in vals_and_setpoints]))
            assert belief


def test_cylindrical_poles(current_driver):
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


def test_spherical_poles(current_driver):
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


def test_warning_increased_max_ramp_rate():
    """
    Test that a warning is raised if we increase the maximum current
    ramp rate. We want the user to be really sure what he or she is
    doing, as this could risk quenching the magnet
    """
    max_ramp_rate = AMI430_VISA._DEFAULT_CURRENT_RAMP_LIMIT
    # Increasing the maximum ramp rate should raise a warning
    target_ramp_rate = max_ramp_rate + 0.01

    with pytest.warns(AMI430Warning,
                      match="Increasing maximum ramp rate") as excinfo:
        inst = AMI430_VISA("testing_increased_max_ramp_rate",
                           address='GPIB::4::INSTR', visalib=visalib,
                           terminator='\n', port=1,
                           current_ramp_limit=target_ramp_rate)
        assert len(excinfo) >= 1  # Check we at least one warning.
        inst.close()


def test_ramp_rate_exception(current_driver):
    """
    Test that an exception is raised if we try to set the ramp rate
    to a higher value than is allowed
    """
    max_ramp_rate = AMI430_VISA._DEFAULT_CURRENT_RAMP_LIMIT
    target_ramp_rate = max_ramp_rate + 0.01
    ix = current_driver._instrument_x

    with pytest.raises(Exception) as excinfo:
        ix.ramp_rate(target_ramp_rate)

        errmsg = "must be between 0 and {} inclusive".format(max_ramp_rate)

        assert errmsg in excinfo.value.args[0]


def test_reducing_field_ramp_limit_reduces_a_higher_ramp_rate(ami430):
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


def test_reducing_current_ramp_limit_reduces_a_higher_ramp_rate(ami430):
    """
    When reducing current_ramp_limit, the actual ramp_rate should also be
    reduced if the new current_ramp_limit is lower than the actual ramp_rate
    now (with respect to field/current conversion).
    """
    factor = 0.8

    # The following fact is expected for the test
    assert ami430.ramp_rate() \
        <= ami430.current_ramp_limit() * ami430.coil_constant()

    # Set ramp_rate_limit to value that is smaller than the ramp_rate of now
    new_current_ramp_limit = ami430.ramp_rate() \
        * factor / ami430.coil_constant()
    ami430.current_ramp_limit(new_current_ramp_limit)

    # Assert that the ramp_rate changed to fit within the new field_ramp_limit
    assert ami430.ramp_rate() <= ami430.field_ramp_limit()
    assert ami430.ramp_rate() \
        <= ami430.current_ramp_limit() * ami430.coil_constant()

    # Well, actually, the new ramp_rate is equal to the new field_ramp_limit
    assert ami430.ramp_rate() == ami430.field_ramp_limit()


def test_reducing_field_ramp_limit_keeps_a_lower_ramp_rate_as_is(ami430):
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


def test_reducing_current_ramp_limit_keeps_a_lower_ramp_rate_as_is(ami430):
    """
    When reducing current_ramp_limit, the actual ramp_rate should remain
    if the new current_ramp_limit is higher than the actual ramp_rate now
    (with respect to field/current conversion).
    """
    factor = 1.2

    # The following fact is expected for the test
    assert ami430.ramp_rate() \
        <= ami430.current_ramp_limit() * ami430.coil_constant()

    old_ramp_rate = ami430.ramp_rate()

    # Set ramp_rate_limit to value that is larger than the ramp_rate of now
    new_current_ramp_limit = ami430.ramp_rate() \
        * factor / ami430.coil_constant()
    ami430.current_ramp_limit(new_current_ramp_limit)

    # Assert that the ramp_rate remained within the new field_ramp_limit
    assert ami430.ramp_rate() <= ami430.field_ramp_limit()
    assert ami430.ramp_rate() \
        <= ami430.current_ramp_limit() * ami430.coil_constant()

    # Assert that ramp_rate hasn't actually changed
    assert ami430.ramp_rate() == old_ramp_rate


def test_blocking_ramp_parameter(current_driver, caplog):

    assert current_driver.block_during_ramp() is True

    log_name = 'qcodes.instrument.base'

    with caplog.at_level(logging.DEBUG, logger=log_name):
        current_driver.cartesian((0, 0, 0))
        caplog.clear()
        current_driver.cartesian((0, 0, 1))

        messages = [record.message for record in caplog.records]
        assert messages[-1] == '[z(AMI430_VISA)] Finished blocking ramp'
        assert messages[-6] == \
            '[z(AMI430_VISA)] Starting blocking ramp of z to 1.0'

        caplog.clear()
        current_driver.block_during_ramp(False)
        current_driver.cartesian((0, 0, 0))
        messages = [record.message for record in caplog.records]

        assert len([mssg for mssg in messages if 'blocking' in mssg]) == 0


def test_current_and_field_params_interlink_at_init(ami430):
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

    np.testing.assert_almost_equal(
        field_ramp_limit, current_ramp_limit*coil_constant)

    np.testing.assert_almost_equal(
        field_limit, current_limit*coil_constant)


def test_current_and_field_params_interlink__change_current_ramp_limit(
        ami430, factor=0.9):
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
    np.testing.assert_almost_equal(
        current_ramp_limit, current_ramp_limit_new)
    np.testing.assert_almost_equal(
        field_ramp_limit, field_ramp_limit_new_expected)

    # The following parameters are not expected to change
    np.testing.assert_almost_equal(
        coil_constant, coil_constant_old)
    np.testing.assert_almost_equal(
        current_limit, current_limit_old)
    np.testing.assert_almost_equal(
        field_limit, field_limit_old)

    # Proportions are expected to hold between field and current parameters
    np.testing.assert_almost_equal(
        field_ramp_limit, current_ramp_limit*coil_constant)
    np.testing.assert_almost_equal(
        field_limit, current_limit*coil_constant)


def test_current_and_field_params_interlink__change_field_ramp_limit(
        ami430, factor=0.9):
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
    np.testing.assert_almost_equal(
        field_ramp_limit, field_ramp_limit_new)
    np.testing.assert_almost_equal(
        current_ramp_limit, current_ramp_limit_new_expected)

    # The following parameters are not expected to change
    np.testing.assert_almost_equal(
        coil_constant, coil_constant_old)
    np.testing.assert_almost_equal(
        current_limit, current_limit_old)
    np.testing.assert_almost_equal(
        field_limit, field_limit_old)

    # Proportions are expected to hold between field and current parameters
    np.testing.assert_almost_equal(
        field_ramp_limit, current_ramp_limit*coil_constant)
    np.testing.assert_almost_equal(
        field_limit, current_limit*coil_constant)


def test_current_and_field_params_interlink__change_coil_constant(
        ami430, factor=3):
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
    np.testing.assert_almost_equal(
        coil_constant, coil_constant_new)
    np.testing.assert_almost_equal(
        current_ramp_limit, current_ramp_limit_new_expected)
    np.testing.assert_almost_equal(
        field_ramp_limit, field_ramp_limit_new_expected)
    np.testing.assert_almost_equal(
        current_limit, current_limit_new_expected)
    np.testing.assert_almost_equal(
        field_limit, field_limit_new_expected)

    # Proportions are expected to hold between field and current parameters
    np.testing.assert_almost_equal(
        field_ramp_limit, current_ramp_limit*coil_constant)
    np.testing.assert_almost_equal(
        field_limit, current_limit*coil_constant)


def test_current_and_field_params_interlink__permutations_of_tests(ami430):
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
        warnings.simplefilter('ignore', category=AMI430Warning)

        test_current_and_field_params_interlink_at_init(ami430)

        test_current_and_field_params_interlink__change_coil_constant(
            ami430, factor=1.2)
        test_current_and_field_params_interlink__change_field_ramp_limit(
            ami430, factor=1.0023)
        test_current_and_field_params_interlink__change_current_ramp_limit(
            ami430, factor=0.98)

        test_current_and_field_params_interlink__change_coil_constant(
            ami430, factor=1.53)
        test_current_and_field_params_interlink__change_current_ramp_limit(
            ami430, factor=2.0)
        test_current_and_field_params_interlink__change_field_ramp_limit(
            ami430, factor=0.633)

        test_current_and_field_params_interlink__change_field_ramp_limit(
            ami430, factor=1.753)
        test_current_and_field_params_interlink__change_coil_constant(
            ami430, factor=0.876)
        test_current_and_field_params_interlink__change_current_ramp_limit(
            ami430, factor=4.6)

        test_current_and_field_params_interlink__change_coil_constant(
            ami430, factor=1.87)
        test_current_and_field_params_interlink__change_current_ramp_limit(
            ami430, factor=2.11)
        test_current_and_field_params_interlink__change_field_ramp_limit(
            ami430, factor=1.0020)

        test_current_and_field_params_interlink__change_field_ramp_limit(
            ami430, factor=0.42)
        test_current_and_field_params_interlink__change_current_ramp_limit(
            ami430, factor=3.1415)
        test_current_and_field_params_interlink__change_coil_constant(
            ami430, factor=1.544)

        test_current_and_field_params_interlink__change_coil_constant(
            ami430, factor=0.12)
        test_current_and_field_params_interlink__change_field_ramp_limit(
            ami430, factor=0.4422)
        test_current_and_field_params_interlink__change_current_ramp_limit(
            ami430, factor=0.00111)


def _parametrization_kwargs():
    kwargs = {'argvalues': [], 'ids': []}

    for type_constructor, type_name in zip(
        ((int, float)
         + numpy_concrete_ints
         + numpy_non_concrete_ints_instantiable
         + numpy_concrete_floats
         + numpy_non_concrete_floats_instantiable),
        (['int', 'float']
         + [str(t) for t in numpy_concrete_ints]
         + [str(t) for t in numpy_non_concrete_ints_instantiable]
         + [str(t) for t in numpy_concrete_floats]
         + [str(t) for t in numpy_non_concrete_floats_instantiable])
    ):
        kwargs['argvalues'].append(type_constructor(2.2))
        kwargs['ids'].append(type_name)

    return kwargs


@pytest.mark.parametrize('field_limit', **_parametrization_kwargs())
def test_numeric_field_limit(magnet_axes_instances, field_limit, request):
    mag_x, mag_y, mag_z = magnet_axes_instances
    ami = AMI430_3D("AMI430-3D", mag_x, mag_y, mag_z, field_limit)
    request.addfinalizer(ami.close)

    assert isinstance(ami._field_limit, float)

    target_within_limit = (field_limit * 0.95, 0, 0)
    ami.cartesian(target_within_limit)

    target_outside_limit = (field_limit * 1.05, 0, 0)
    with pytest.raises(ValueError,
                       match='_set_fields aborted; field would exceed limit'):
        ami.cartesian(target_outside_limit)
