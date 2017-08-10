"""
A convenient class to keep track of vectors representing physical fields. The idea is that a vector instance
stores a representation in cartesian, spherical and cylindrical coordinates. Giving either (x, y, z) values or
(rho, phi, z) values or (r, theta, phi) values at instantiation will calculate the other representation immediately.
"""

import numpy as np


class FieldVector(object):
    attributes = ["x", "y", "z", "r", "theta", "phi", "rho"]

    def __init__(self, x=None, y=None, z=None, r=None, theta=None, phi=None, rho=None):
        """
        All inputs are optional, however the user needs to either give (x, y, z) values,
        (r, theta, phi) values or (phi, rho, z) values. It is in principle possible to
        solve for all parameters if, for instance x, y, theta are known (this is a uniquely
        defined coordinate), however, it is difficult to write elegant code to generalize this.
        """

        self._x = x
        self._y = y
        self._z = z

        self._r = r
        self._theta = theta
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
                raise ValueError("Computed value of {} inconsistent with given value".format(attr_name))

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
        theta = np.arccos(z / r)

        return x, y, z, r, theta, phi, rho

    def _compute_unknowns(self):
        """
        Compute all values. This function will raise an error if:
         1) There are contradictory inputs (e.g. x=3, y=4, z=0 and rho=6)
        """

        for _ in range(1):

            new_values = FieldVector._cartesian_to_other(self._x, self._y, self._z)
            if new_values is not None:
                self._set_attribute_values(FieldVector.attributes, new_values)
                break

            new_values = FieldVector._spherical_to_other(self._r, self._theta, self._phi)
            if new_values is not None:
                self._set_attribute_values(FieldVector.attributes, new_values)
                break

            new_values = FieldVector._cylindrical_to_other(self._phi, self._rho, self._z)
            if new_values is not None:
                self._set_attribute_values(FieldVector.attributes, new_values)
                break

    def copy(self, other):
        """Copy the properties of other vector to yourself"""
        for att in FieldVector.attributes:
            value = getattr(other, "_"+ att)
            setattr(self, "_" + att, value)

    def set_vector(self, **new_values):

        names = sorted(list(new_values.keys()))
        groups = [["x", "y", "z"], ["phi", "r", "theta"], ["phi", "rho", "z"]]
        if names not in groups:
            raise ValueError("Can only set vector with a complete value set")

        new_vector = FieldVector(**new_values)
        self.copy(new_vector)

    def set_component(self, **new_values):
        """
        Set components of the vector to some new value. It is disallowed for the user to set vector components
        manually as this can lead to inconsistencies (e.g. x and rho are not independent of each other, setting
        one has to effect the other)

        :param new_values: dict, keys representing parameter names and values the values to be set
        """

        if len(new_values) > 1:
            raise NotImplementedError("Cannot set multiple components at once")

        items = list(new_values.items())
        component_name = items[0][0]
        value = items[0][1]

        setattr(self, "_" + component_name, value)
        # Setting x (for example), we will keep y and z untouched, but set r, theta, phi and rho to None
        # so these can be recomputed. This will keep things consistent.
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
        return [getattr(self, "_" + name) for name in names]

    # A method property allows us to confidently retrieve values but disallows modifying them
    # We can do this:
    # a = field_vector.x
    # But this:
    # field_vector.x = a
    # Will raise an error.

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
        return self._theta

    @property
    def r(self):
        return self._r

    @property
    def phi(self):
        return self._phi
