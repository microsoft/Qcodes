"""
A convenient class to keep track of vectors representing physical fields. The
idea is that a vector instance stores a representation in cartesian, spherical
and cylindrical coordinates. Giving either (x, y, z) values or (rho, phi, z)
values or (r, theta, phi) values at instantiation we will calculate the other
representation immediately.
"""

import numpy as np


class FieldVector(object):
    attributes = ["x", "y", "z", "r", "theta", "phi", "rho"]

    def __init__(self, x=None, y=None, z=None, r=None, theta=None, phi=None,
                 rho=None):
        """
        Parameters:
            x (float, optional): represents the norm of the projection of the
                                    vector along the x-axis
            y (float, optional): represents the norm of the projection of the
                                    vector along the y-axis
            z (float, optional): represents the norm of the projection of the
                                    vector along the z-axis
            r (float, optional): represents the norm of the vector
            theta (float, optional): represents the angle of the vector with
                                        respect to the positive z-axis
            rho (float, optional): represents the norm of the projection of the
                                    vector on to the xy-plane
            phi (float, optional): represents the angle of rho with respect to
                                        the positive x-axis

        Note: All inputs are optional, however the user needs to either give
                (x, y, z) values, (r, theta, phi) values or (phi, rho, z)
                values for meaningful computation
        """
        self._x = x
        self._y = y
        self._z = z

        self._r = r
        if theta is not None:
            self._theta = np.radians(theta)
        else:
            self._theta = theta

        if phi is not None:
            self._phi = np.radians(phi)
        else:
            self._phi = phi

        self._rho = rho

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

    @staticmethod
    def _cartesian_to_other(x, y, z):
        """ Convert a cartesian set of coordinates to values of interest."""

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
        """Convert from spherical to other representations"""

        if any([i is None for i in [r, theta, phi]]):
            return None

        z = r * np.cos(theta)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        rho = np.sqrt(x ** 2 + y ** 2)

        return x, y, z, r, theta, phi, rho

    @staticmethod
    def _cylindrical_to_other(phi, rho, z):
        """Convert from cylindrical to other representations"""

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
        to contain no None values, or the set (r, theta, phi), or the set
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

    def copy(self, other):
        """Copy the properties of other vector to yourself"""
        for att in FieldVector.attributes:
            value = getattr(other, "_" + att)
            setattr(self, "_" + att, value)

    def set_vector(self, **new_values):
        """
        Reset the the values of the vector

        Examples:
            >>> f = FieldVector(x=0, y=2, z=6)
            >>> f.set_vector(x=9, y=3, z=1)
            >>> f.set_vector(r=1, theta=30.0, phi=10.0)
            >>> f.set_vector(x=9, y=0)  # this should raise a value error:
            # "Can only set vector with a complete value set"
            >>> f.set_vector(x=9, y=0, r=3)  # although mathematically it is
            # possible to compute the complete vector from the values given,
            # this is too hard to implement with generality (and not worth it)
            # so this to will raise the above mentioned ValueError
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
        other, setting one has to effect the other)

        Examples:
            >>> f = FieldVector(x=2, y=3, z=4)
            >>> f.set_component(r=10) # Since r is part of the set
            # (r, theta, phi) representing spherical coordinates, setting r
            # means that theta and phi are kept constant and only r is changed.
            # After changing r, (x, y, z) values are recomputed, as is the rho
            # coordinate. Internally we arrange this by setting x, y, z and
            # rho to None and calling self._compute_unknowns()

        Parameters:
            new_values (dict): keys representing parameter names and values the
            values to be set
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

        setattr(self, "_" + component_name, value)

        groups = [["x", "y", "z"], ["r", "theta", "phi"], ["phi", "rho", "z"]]

        for group in groups:
            if component_name in group:

                for att in FieldVector.attributes:
                    if att not in group:
                        setattr(self, "_" + att, None)

                break

        self._compute_unknowns()

    def get_components(self, *names):
        """Get field components by name"""

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
        Returns True if other is equivalent to self, False otherwise
        """
        for name in ["x", "y", "z"]:
            self_value = getattr(self, name)
            other_value = getattr(other, name)
            if not np.isclose(self_value, other_value):
                return False

        return True

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def rho(self):
        return self._rho

    @property
    def theta(self):
        return np.degrees(self._theta)

    @property
    def r(self):
        return self._r

    @property
    def phi(self):
        return np.degrees(self._phi)
