import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import floats
from hypothesis.strategies import tuples

import qcodes.instrument.sims as sims
from qcodes.instrument_drivers.american_magnetics.AMI430 import AMI430_3D
from qcodes.instrument_drivers.american_magnetics.AMI430_VISA import AMI430_VISA
from qcodes.math.field_vector import FieldVector

# If any of the field limit functions are satisfied we are in the safe zone.
# We can have higher field along the z-axis if x and y are zero.
field_limit = [
    lambda x, y, z: x == 0 and y == 0 and z < 3,
    lambda x, y, z: np.linalg.norm([x, y, z]) < 2
]

# path to the .yaml file containing the simulated instrument
visalib = sims.__file__.replace('__init__.py', 'AMI430.yaml@sim')


@pytest.fixture(scope='module')
def current_driver():
    """
    Start three mock instruments representing current drivers for the x, y,
    and z directions.
    """

    mag_x = AMI430_VISA('x', address='GPIB::1::65535::INSTR', visalib=visalib,
                        terminator='\n', port=1)
    mag_y = AMI430_VISA('y', address='GPIB::2::65535::INSTR', visalib=visalib,
                        terminator='\n', port=1)
    mag_z = AMI430_VISA('z', address='GPIB::3::65535::INSTR', visalib=visalib,
                        terminator='\n', port=1)

    driver = AMI430_3D("AMI430-3D", mag_x, mag_y, mag_z, field_limit)

    return driver


# here the original test has a log system that we probably don't want to
# reproduce / write tests for

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


@given(random_coordinates["cartesian"])
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


@given(random_coordinates["spherical"])
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


@given(random_coordinates["cylindrical"])
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


@given(random_coordinates["cartesian"])
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


@given(random_coordinates["spherical"])
@settings(max_examples=10)
def test_spherical_setpoints(current_driver, set_target):
    """
    Check that the individual x, y, z instruments are getting the set
    points as intended. This test is very similar to the sanity test, but
    adds in the FieldVector as well.
    """

    print(set_target)

    current_driver.spherical(set_target)

    r = current_driver.field()
    theta = current_driver.theta()
    phi = current_driver.phi()

    get_target = dict(zip(('r', 'theta', 'phi'), (r, theta, phi)))
    set_target = dict(zip(('r', 'theta', 'phi'), set_target))

    set_vector = FieldVector(**set_target)
    get_vector = FieldVector(**get_target)
    assert set_vector.is_equal(get_vector)


@given(random_coordinates["cylindrical"])
@settings(max_examples=10)
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


@given(random_coordinates["cartesian"])
@settings(max_examples=10)
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

    assert FieldVector(*set_target).is_equal(FieldVector(x=cartesian_x,
                                                         y=cartesian_y,
                                                         z=cartesian_z))
    assert np.allclose(cartesian, [cartesian_x, cartesian_y, cartesian_z])

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
