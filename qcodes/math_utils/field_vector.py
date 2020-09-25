"""
A helper module containing a class to keep track of vectors in different
coordinate systems.
"""


import numpy as np

from typing import Union, Type, TypeVar, Optional
NormOrder = Union[str, float]
T = TypeVar('T', bound='FieldVector')


class FieldVector:
    """
    A convenient class to keep track of vectors representing physical fields.
    The idea is that a vector instance stores a representation in Cartesian,
    spherical and cylindrical coordinates. All arguments are optional, however
    the user needs to provide one of the combinations of either (x, y, z) values
    or (rho, phi, z) values or (r, theta, phi) values at instantiation for a
    meaningful computation of the other representation, immediately.
    """
    attributes = ["x", "y", "z", "r", "theta", "phi", "rho"]
    repr_format = "cartesian"

    def __init__(self,
                 x: Optional[float] = None,
                 y: Optional[float] = None,
                 z: Optional[float] = None,
                 r: Optional[float] = None,
                 theta: Optional[float] = None,
                 phi: Optional[float] = None,
                 rho: Optional[float] = None):
        """
        Args:
            x: represents the norm of the projection
                of the vector along the x-axis
            y: represents the norm of the projection
                of the vector along the y-axis
            z: represents the norm of the projection
                of the vector along the z-axis
            r: represents the norm of the vector
            theta: represents the angle of the vector
                with respect to the positive z-axis
            rho: represents the norm of the projection
                of the vector on to the xy-plane
            phi: represents the angle of rho
                with respect to the positive x-axis
        """

        self._x = float(x) if x is not None else None
        self._y = float(y) if y is not None else None
        self._z = float(z) if z is not None else None

        self._r = float(r) if r is not None else None
        self._theta = float(np.radians(theta)) if theta is not None else None
        self._phi = float(np.radians(phi)) if phi is not None else None
        self._rho = float(rho) if rho is not None else None

        self._compute_unknowns()

    def _set_attribute_value(self, attr_name, value):
        if value is None:
            return

        attr_value = getattr(self, "_" + attr_name)

        if attr_value is None:
            setattr(self, "_" + attr_name, value)
        else:
            if not np.isclose(attr_value, value):
                raise ValueError(
                    f"Computed value of {attr_name} inconsistent with given "
                    f"value"
                )

    def _set_attribute_values(self, attr_names, values):
        for attr_name, value in zip(attr_names, values):
            self._set_attribute_value(attr_name, value)

    def __getnewargs__(self):
        return self.x, self.y, self.z

    @staticmethod
    def _cartesian_to_other(x, y, z):
        """Convert a cartesian set of coordinates to values of interest."""
        if any([i is None for i in [x, y, z]]):
            return None

        phi = np.arctan2(y, x)
        rho = np.sqrt(x ** 2 + y ** 2)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if r != 0:
            theta = np.arccos(z / r)
        else:
            theta = 0

        return x, y, z, r, theta, phi, rho

    @staticmethod
    def _spherical_to_other(r, theta, phi):
        """Convert from spherical to other representations."""
        if any([i is None for i in [r, theta, phi]]):
            return None

        z = r * np.cos(theta)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        rho = np.sqrt(x ** 2 + y ** 2)

        return x, y, z, r, theta, phi, rho

    @staticmethod
    def _cylindrical_to_other(phi, rho, z):
        """Convert from cylindrical to other representations."""
        if any([i is None for i in [phi, rho, z]]):
            return None

        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        r = np.sqrt(rho ** 2 + z ** 2)
        if r != 0:
            theta = np.arccos(z / r)
        else:
            theta = 0

        return x, y, z, r, theta, phi, rho

    def _compute_unknowns(self):
        """
        Compute all coordinates. To do this we need either the set (x, y, z)
        to contain no ``None`` values, or the set (r, theta, phi), or the set
        (rho, phi, z). Given any of these sets, we can recompute the rest.

        This function will raise an error if there are contradictory inputs
        (e.g. x=3, y=4, z=0 and rho=6).
        """
        for f in [
            lambda: FieldVector._cartesian_to_other(self._x, self._y, self._z),
            lambda: FieldVector._spherical_to_other(self._r, self._theta,
                                                    self._phi),
            lambda: FieldVector._cylindrical_to_other(self._phi, self._rho,
                                                      self._z)
        ]:
            new_values = f()
            if new_values is not None:  # this will return None if any of the
                # function arguments is None.
                self._set_attribute_values(FieldVector.attributes, new_values)
                break

    def copy(self: T, other: T):
        """Copy the properties of other vector to yourself."""
        for att in FieldVector.attributes:
            value = getattr(other, "_" + att)
            setattr(self, "_" + att, value)

    def set_vector(self, **new_values):
        """
        Reset the the values of the vector.

        Examples:
            >>> f = FieldVector(x=0, y=2, z=6)
            >>> f.set_vector(x=9, y=3, z=1)
            >>> f.set_vector(r=1, theta=30.0, phi=10.0)
            # The following should raise a value error:
            # "Can only set vector with a complete value set"
            >>> f.set_vector(x=9, y=0)
            # Although mathematically it is possible to compute the complete
            # vector from the values given, this is too hard to implement with
            # generality (and not worth it), so the following will raise the
            # above-mentioned ValueError too.
            >>> f.set_vector(x=9, y=0, r=3)
        """
        names = sorted(list(new_values.keys()))
        groups = [["x", "y", "z"], ["phi", "r", "theta"], ["phi", "rho", "z"]]
        if names not in groups:
            raise ValueError("Can only set vector with a complete value set")

        new_vector = FieldVector(**new_values)
        self.copy(new_vector)

    def set_component(self, **new_values):
        """
        Set a single component of the vector to some new value. It is
        disallowed for the user to set vector components manually as this can
        lead to inconsistencies (e.g. x and rho are not independent of each
        other, setting one has to effect the other).

        Examples:
            >>> f = FieldVector(x=2, y=3, z=4)
            # Since r is part of the set (r, theta, phi) representing
            # spherical coordinates, setting r means that theta and phi are
            # kept constant and only r is changed. After changing r,
            # (x, y, z) values are recomputed, as is the rho coordinate.
            # Internally we arrange this by setting x, y, z and rho to None
            # and calling self._compute_unknowns().
            >>> f.set_component(r=10)

        Args:
            new_values (dict): Keys representing parameter names and values the
                values to be set.
        """
        if len(new_values) > 1:
            raise NotImplementedError("Cannot set multiple components at once")

        items = list(new_values.items())
        component_name = items[0][0]

        if component_name in ["theta", "phi"]:
            # convert angles to radians
            value = np.radians(items[0][1])
        else:
            value = items[0][1]

        setattr(self, "_" + component_name, float(value))

        groups = [["x", "y", "z"], ["r", "theta", "phi"], ["phi", "rho", "z"]]

        for group in groups:
            if component_name in group:

                for att in FieldVector.attributes:
                    if att not in group:
                        setattr(self, "_" + att, None)

                break

        self._compute_unknowns()

    def get_components(self, *names):
        """Get field components by name."""

        def convert_angle_to_degrees(name, value):
            # Convert all angles to degrees
            if name in ["theta", "phi"]:
                return np.degrees(value)
            else:
                return value

        components = [convert_angle_to_degrees(
            name, getattr(self, "_" + name)
        ) for name in names]

        return components

    def is_equal(self, other):
        """
        Returns ``True`` if ``other`` is equivalent to ``self``, ``False`` otherwise.
        """
        for name in ["x", "y", "z"]:
            self_value = getattr(self, name)
            other_value = getattr(other, name)
            if not np.isclose(self_value, other_value):
                return False

        return True

    def __getitem__(self, component):
        return self.get_components(component)[0]

    def __setitem__(self, component, value):
        self.set_component(**{component: value})

    def __mul__(self, other):
        if not isinstance(other, (float, int)):
            return NotImplemented

        return FieldVector(**{
            component: self[component] * other
            for component in 'xyz'
        })

    def __rmul__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented

        return self * other

    def __neg__(self):
        return -1 * self

    def __add__(self, other):
        if not isinstance(other, FieldVector):
            return NotImplemented

        return FieldVector(**{
            component: self[component] + other[component]
            for component in 'xyz'
        })

    def __sub__(self, other):
        if not isinstance(other, FieldVector):
            return NotImplemented

        return FieldVector(**{
            component: self[component] - other[component]
            for component in 'xyz'
        })

    # NB: we disable the pylint warning here so that we can match
    #     NumPy's naming convention for the norm method.
    def norm(self,
             ord: NormOrder = 2  # pylint: disable=redefined-builtin
             ) -> float:
        """
        Returns the norm of this field vector. See ``np.norm``
        for the definition of the ord keyword argument.
        """
        return np.linalg.norm([self.x, self.y, self.z], ord=ord)

    def distance(self, other,
                 ord: NormOrder = 2  # pylint: disable=redefined-builtin
                 ) -> float:
        return (self - other).norm(ord=ord)

    @property
    def x(self) -> Optional[float]:
        return self._x

    @property
    def y(self) -> Optional[float]:
        return self._y

    @property
    def z(self) -> Optional[float]:
        return self._z

    @property
    def rho(self) -> Optional[float]:
        return self._rho

    @property
    def theta(self) -> Optional[float]:
        return float(np.degrees(self._theta))

    @property
    def r(self) -> Optional[float]:
        return self._r

    @property
    def phi(self) -> Optional[float]:
        return float(np.degrees(self._phi))

    # Representation Methods #

    def repr_cartesian(self) -> str:
        return f"FieldVector(x={self.x}, y={self.y}, z={self.z})"

    def repr_spherical(self) -> str:
        return f"FieldVector(r={self.r}, phi={self.phi}, theta={self.theta})"

    def repr_cylindrical(self) -> str:
        return f"FieldVector(rho={self.rho}, phi={self.phi}, z={self.z})"

    def __repr__(self) -> str:
        if self.repr_format == "cartesian":
            return self.repr_cartesian()
        elif self.repr_format == "spherical":
            return self.repr_spherical()
        elif self.repr_format == "cylindrical":
            return self.repr_cylindrical()
        else:
            return super().__repr__()

    # Homogeneous Coordinates #

    def as_homogeneous(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, 1])

    @classmethod
    def from_homogeneous(cls: Type[T], hvec: np.ndarray) -> T:
        # Homogeneous coordinates define an equivalence relation
        #     [x / s, y / s, z / s, 1] == [x, y, z, s].
        # More generally,
        #     [x, y, z, s] == [x', y', z', s']
        #     iff x / s == x' / s',
        #         y / s == y' / s', and
        #         z / s == z' / s'.
        # This definition has the consequence that for any w,
        # w * [x, y, z, s] == [x, y, z, s].
        # Thus, we start by rescaling such that s == 1.
        hvec /= hvec[-1]
        return cls(
            x=hvec[0], y=hvec[1], z=hvec[2]
        )
