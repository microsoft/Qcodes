"""
A helper module containing a class to keep track of vectors in different
coordinate systems.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar, Union

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

AllCoordsType = tuple[float, float, float, float, float, float, float]
NormOrder = Union[None, float, Literal["fro"], Literal["nuc"]]
T = TypeVar("T", bound="FieldVector")


class FieldVector:
    """
    A convenient class to keep track of vectors representing physical fields.
    The idea is that a vector instance stores a representation in Cartesian,
    spherical and cylindrical coordinates. All arguments are optional, however
    the user needs to provide one of the combinations of either (x, y, z) values
    or (rho, phi, z) values or (r, theta, phi) values at instantiation for a
    meaningful computation of the other representation, immediately.
    """

    attributes: ClassVar[list[str]] = ["x", "y", "z", "r", "theta", "phi", "rho"]
    repr_format = "cartesian"

    def __init__(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        r: float | None = None,
        theta: float | None = None,
        phi: float | None = None,
        rho: float | None = None,
    ):
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

    def _set_attribute_value(self, attr_name: str, value: float | None) -> None:
        if value is None:
            return

        attr_value = getattr(self, "_" + attr_name)

        if attr_value is None:
            setattr(self, "_" + attr_name, value)
        elif not np.isclose(attr_value, value):
            raise ValueError(
                f"Computed value of {attr_name} inconsistent with given value"
            )

    def _set_attribute_values(
        self, attr_names: Sequence[str], values: Sequence[float | None]
    ) -> None:
        for attr_name, value in zip(attr_names, values):
            self._set_attribute_value(attr_name, value)

    def __getnewargs__(self) -> tuple[float | None, float | None, float | None]:
        return self.x, self.y, self.z

    @staticmethod
    def _cartesian_to_other(
        x: float | None, y: float | None, z: float | None
    ) -> AllCoordsType | None:
        """Convert a cartesian set of coordinates to values of interest."""
        if x is None or y is None or z is None:
            return None
        phi = np.arctan2(y, x)
        rho = np.sqrt(x ** 2 + y ** 2)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if r != 0:
            z_r_frac = z / r
            # it it possible that z_r_frac is slightly larger than 1 or
            # slightly smaller than -1 due to floating point errors.
            # an example that triggers this is:
            # x=0, y=0, z=3.729170476738041e-155
            if z_r_frac > 1:
                z_r_frac = 1
            elif z_r_frac < -1:
                z_r_frac = -1
            theta = np.arccos(z_r_frac)
        else:
            theta = 0

        return x, y, z, r, theta, phi, rho

    @staticmethod
    def _spherical_to_other(
        r: float | None, theta: float | None, phi: float | None
    ) -> AllCoordsType | None:
        """Convert from spherical to other representations."""
        if r is None or theta is None or phi is None:
            return None
        z = r * np.cos(theta)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        rho = np.sqrt(x ** 2 + y ** 2)

        return x, y, z, r, theta, phi, rho

    @staticmethod
    def _cylindrical_to_other(
        phi: float | None, rho: float | None, z: float | None
    ) -> AllCoordsType | None:
        """Convert from cylindrical to other representations."""
        if phi is None or rho is None or z is None:
            return None
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        r = np.sqrt(rho ** 2 + z ** 2)
        if r != 0:
            z_r_frac = z / r
            # it it possible that z_r_frac is slightly larger than 1 or
            # slightly smaller than -1 due to floating point errors.
            # an example that triggers this is:
            # phi=0, rho=0, z=3.729170476738041e-155
            if z_r_frac > 1:
                z_r_frac = 1
            elif z_r_frac < -1:
                z_r_frac = -1
            theta = np.arccos(z_r_frac)
        else:
            theta = 0

        return x, y, z, r, theta, phi, rho

    def _compute_unknowns(self) -> None:
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

    def copy(self: T, other: T) -> None:
        """Copy the properties of other vector to yourself."""
        for att in FieldVector.attributes:
            value = getattr(other, "_" + att)
            setattr(self, "_" + att, value)

    def set_vector(self, **new_values: float) -> None:
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

        new_vector = self.__class__(**new_values)
        self.copy(new_vector)

    def set_component(self, **new_values: float) -> None:
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

    def get_components(self, *names: str) -> list[float]:
        """Get field components by name."""

        def convert_angle_to_degrees(name: str, value: float) -> float:
            # Convert all angles to degrees
            if name in ["theta", "phi"]:
                return float(np.degrees(value))
            else:
                return value

        components = [convert_angle_to_degrees(
            name, getattr(self, "_" + name)
        ) for name in names]

        return components

    def is_equal(self, other: FieldVector) -> bool:
        """
        Returns ``True`` if ``other`` is equivalent to ``self``, ``False`` otherwise.
        """
        for name in ["x", "y", "z"]:
            self_value = getattr(self, name)
            other_value = getattr(other, name)
            if not np.isclose(self_value, other_value):
                return False

        return True

    def __getitem__(self, component: str) -> float:
        return self.get_components(component)[0]

    def __setitem__(self, component: str, value: float) -> None:
        self.set_component(**{component: value})

    def __mul__(self, other: Any) -> FieldVector:
        if not isinstance(other, (float, int)):
            return NotImplemented

        return FieldVector(**{
            component: self[component] * other
            for component in 'xyz'
        })

    def __rmul__(self, other: Any) -> FieldVector:
        if not isinstance(other, (int, float)):
            return NotImplemented

        return self * other

    def __truediv__(self, other: Any) -> FieldVector:
        if not isinstance(other, (int, float)):
            return NotImplemented

        return self * (1.0 / other)

    def __neg__(self) -> FieldVector:
        return -1 * self

    def __add__(self, other: Any) -> FieldVector:
        if not isinstance(other, FieldVector):
            return NotImplemented

        return FieldVector(**{
            component: self[component] + other[component]
            for component in 'xyz'
        })

    def __sub__(self, other: Any) -> FieldVector:
        if not isinstance(other, FieldVector):
            return NotImplemented

        return FieldVector(**{
            component: self[component] - other[component]
            for component in 'xyz'
        })

    def norm(self, ord: NormOrder = 2) -> float:
        """
        Returns the norm of this field vector. See ``np.norm``
        for the definition of the ord keyword argument.
        """
        assert self.x is not None
        assert self.y is not None
        assert self.z is not None

        return float(np.linalg.norm([self.x, self.y, self.z], ord=ord))

    def distance(
        self,
        other: FieldVector,
        ord: NormOrder = 2,
    ) -> float:
        return (self - other).norm(ord=ord)

    @property
    def x(self) -> float | None:
        return self._x

    @property
    def y(self) -> float | None:
        return self._y

    @property
    def z(self) -> float | None:
        return self._z

    @property
    def rho(self) -> float | None:
        return self._rho

    @property
    def theta(self) -> float | None:
        if self._theta is None:
            return None
        return float(np.degrees(self._theta))

    @property
    def r(self) -> float | None:
        return self._r

    @property
    def phi(self) -> float | None:
        if self._phi is None:
            return None
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
    def from_homogeneous(cls: type[T], hvec: np.ndarray) -> T:
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
