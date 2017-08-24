"""
Test properties of the coordinate transforms in the field vector module to test if everything has been correctly
implemented.
"""
import numpy as np

from qcodes.math.field_vector import FieldVector


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


def test_spherical_properties():
    """
    If the cartesian to spherical transform has been correctly implemented then we expect certain symmetrical properties
    """
    # Generate a random coordinate in spherical representation
    spherical0 = {name: get_random_coordinate(name) for name in ["r", "theta", "phi"]}
    cartisian0 = FieldVector(**spherical0).get_components("x", "y", "z")

    # Mirror the theta angle in the xy-plane. This should flip the sign of the z-coordinate
    spherical1 = dict(spherical0)
    spherical1["theta"] = 180 - spherical0["theta"]
    cartisian1 = FieldVector(**spherical1).get_components("x", "y", "z")
    assert np.allclose(cartisian0, cartisian1 * np.array([1, 1, -1]))

    # Add 180 to the phi coordinate. This should flip the sign of the xy coordinate
    spherical2 = dict(spherical0)
    spherical2["phi"] = 180 + spherical0["phi"]
    cartisian2 = FieldVector(**spherical2).get_components("x", "y", "z")
    assert np.allclose(cartisian0, cartisian2 * np.array([-1, -1, 1]))

    # Mirroring the theta angle in the xy-plane and adding 180 to the phi coordinate should flip all cartesian
    # coordinates
    spherical3 = dict(spherical0)
    spherical3["theta"] = 180 - spherical0["theta"]  # This should only flip the z-coordinate
    spherical3["phi"] = 180 + spherical0["phi"]  # This should flip the xy-coordinate
    cartisian3 = FieldVector(**spherical3).get_components("x", "y", "z")
    assert np.allclose(cartisian0, cartisian3 * np.array([-1, -1, -1]))

    # Finally flipping the sign of the r coordinate should flip all cartesian coordinates
    spherical4 = dict(spherical0)
    spherical4["r"] = -spherical0["r"]  # This should flip all coordinates
    cartisian4 = FieldVector(**spherical4).get_components("x", "y", "z")
    assert np.allclose(cartisian0, cartisian4 * np.array([-1, -1, -1]))


def test_cylindrical_properties():
    """
    If the cartesian to cylindrical transform has been correctly implemented then we expect certain symmetrical
    properties
    """
    # Generate a random coordinate in cylindrical representation
    cylindrical0 = {name: get_random_coordinate(name) for name in ["rho", "phi", "z"]}
    cartisian0 = FieldVector(**cylindrical0).get_components("x", "y", "z")

    # If we flip the sign of the rho coordinate, we will flip the xy coordinate
    cylindrical1 = dict(cylindrical0)
    cylindrical1["rho"] *= -1
    cartisian1 = FieldVector(**cylindrical1).get_components("x", "y", "z")

    assert np.allclose(cartisian0, cartisian1 * np.array([-1, -1, 1]))
