"""
Test properties of the coordinate transforms in the field vector module to
test if everything has been correctly implemented.
"""
import numpy as np
import json
from hypothesis import given, settings
from hypothesis.strategies import floats
from hypothesis.strategies import tuples

from qcodes.math_utils.field_vector import FieldVector
from qcodes.utils.helpers import NumpyJSONEncoder

random_coordinates = {
    "cartesian": tuples(
        floats(min_value=0, max_value=1),  # x
        floats(min_value=0, max_value=1),  # y
        floats(min_value=0, max_value=1)   # z
    ),
    "spherical": tuples(
        floats(min_value=0, max_value=1),    # r
        floats(min_value=0, max_value=180),  # theta
        floats(min_value=0, max_value=180)   # phi
    ),
    "cylindrical": tuples(
        floats(min_value=0, max_value=1),    # rho
        floats(min_value=0, max_value=180),  # phi
        floats(min_value=0, max_value=1)     # z
    )
}


@given(random_coordinates["spherical"])
@settings(max_examples=10)
def test_spherical_properties(spherical0):
    """
    If the cartesian to spherical transform has been correctly implemented
    then we expect certain symmetrical properties
    """
    # Generate a random coordinate in spherical representation
    spherical0 = dict(zip(["r", "theta", "phi"], spherical0))
    cartisian0 = FieldVector(**spherical0).get_components("x", "y", "z")

    # Mirror the theta angle in the xy-plane.
    # This should flip the sign of the z-coordinate
    spherical1 = dict(spherical0)
    spherical1["theta"] = 180 - spherical0["theta"]
    cartisian1 = FieldVector(**spherical1).get_components("x", "y", "z")
    assert np.allclose(cartisian0, cartisian1 * np.array([1, 1, -1]))

    # Add 180 to the phi coordinate.
    # This should flip the sign of the xy coordinate
    spherical2 = dict(spherical0)
    spherical2["phi"] = 180 + spherical0["phi"]
    cartisian2 = FieldVector(**spherical2).get_components("x", "y", "z")
    assert np.allclose(cartisian0, cartisian2 * np.array([-1, -1, 1]))

    # Mirroring the theta angle in the xy-plane
    # and adding 180 to the phi coordinate
    # should flip all cartesian coordinates
    spherical3 = dict(spherical0)
    spherical3["theta"] = 180 - spherical0["theta"]  # should only flip z
    spherical3["phi"] = 180 + spherical0["phi"]  # should flip xy
    cartisian3 = FieldVector(**spherical3).get_components("x", "y", "z")
    assert np.allclose(cartisian0, cartisian3 * np.array([-1, -1, -1]))

    # Finally, flipping the sign of the r coordinate
    # should flip all cartesian coordinates
    spherical4 = dict(spherical0)
    spherical4["r"] = -spherical0["r"]  # This should flip all coordinates
    cartisian4 = FieldVector(**spherical4).get_components("x", "y", "z")
    assert np.allclose(cartisian0, cartisian4 * np.array([-1, -1, -1]))


@given(random_coordinates["cylindrical"])
@settings(max_examples=10)
def test_cylindrical_properties(cylindrical0):
    """
    If the cartesian to cylindrical transform has been correctly implemented
    then we expect certain symmetrical properties
    """
    # Generate a random coordinate in cylindrical representation
    cylindrical0 = dict(zip(["rho", "phi", "z"], cylindrical0))
    cartisian0 = FieldVector(**cylindrical0).get_components("x", "y", "z")

    # If we flip the sign of the rho coordinate, we will flip the xy coordinate
    cylindrical1 = dict(cylindrical0)
    cylindrical1["rho"] *= -1
    cartisian1 = FieldVector(**cylindrical1).get_components("x", "y", "z")

    assert np.allclose(cartisian0, cartisian1 * np.array([-1, -1, 1]))


@given(
    random_coordinates["cylindrical"],
    random_coordinates["spherical"]
)
@settings(max_examples=10)
def test_triangle_inequality(cylindrical0, spherical0):
    cylindrical0 = FieldVector(**dict(zip(["rho", "phi", "z"], cylindrical0)))
    spherical0 = FieldVector(**dict(zip(["r", "phi", "theta"], spherical0)))

    assert (cylindrical0 + spherical0).norm() \
           <= (cylindrical0.norm() + spherical0.norm())
    assert cylindrical0.distance(spherical0) \
           <= (cylindrical0.norm() + spherical0.norm())


@given(random_coordinates["cartesian"])
@settings(max_examples=10)
def test_homogeneous_roundtrip(cartesian0):
    vec = FieldVector(**dict(zip("xyz", cartesian0)))
    h_vec = 13 * vec.as_homogeneous()

    assert np.allclose(
        vec.get_components(*"xyz"),
        FieldVector.from_homogeneous(h_vec).get_components(*"xyz")
    )


@given(random_coordinates["spherical"])
@settings(max_examples=10)
def test_json_dump(spherical0):
    vec = FieldVector(**dict(zip(["r", "phi", "theta"], spherical0)))
    dump = json.dumps(vec, cls=NumpyJSONEncoder)

    assert json.loads(dump) == {
        '__class__': FieldVector.__name__,
        '__args__': [vec.x, vec.y, vec.z]
    }


def test_all_attributes_are_floats():
    cartesian0 = (400, 200, 300)
    cylindrical0 = (1, 52, 0)
    spherical0 = (1, 78, 145)

    cartesian = FieldVector(**dict(zip("xyz", cartesian0)))
    cylindrical = FieldVector(**dict(zip(["rho", "phi", "z"], cylindrical0)))
    spherical = FieldVector(**dict(zip(["r", "phi", "theta"], spherical0)))

    # Test that all attributes are floats upon creation
    for fv in [cartesian, cylindrical, spherical]:
        for attr in FieldVector.attributes:
            assert isinstance(getattr(fv, attr), float)

    # Test that all attributes are floats even after setting components
    for fv in [cartesian, cylindrical, spherical]:
        for set_comp in FieldVector.attributes:
            fv.set_component(**{set_comp: 1})

            for attr in FieldVector.attributes:
                assert isinstance(getattr(fv, attr), float)
