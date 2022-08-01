from contextlib import contextmanager
from typing import Any, Iterator, List, Sequence, Tuple, Union


class DelegateAttributes:
    """
    Mixin class to create attributes of this object by
    delegating them to one or more dictionaries and/or objects.

    Also fixes ``__dir__`` so the delegated attributes will show up
    in ``dir()`` and ``autocomplete``.

    Attribute resolution order:
        1. Real attributes of this object.
        2. Keys of each dictionary in ``delegate_attr_dicts`` (in order).
        3. Attributes of each object in ``delegate_attr_objects`` (in order).
    """

    delegate_attr_dicts: List[str] = []
    """
    A list of names (strings) of dictionaries
    which are (or will be) attributes of ``self``, whose keys should
    be treated as attributes of ``self``.
    """
    delegate_attr_objects: List[str] = []
    """
    A list of names (strings) of objects
    which are (or will be) attributes of ``self``, whose attributes
    should be passed through to ``self``.
    """
    omit_delegate_attrs: List[str] = []
    """
    A list of attribute names (strings)
    to *not* delegate to any other dictionary or object.
    """

    def __getattr__(self, key: str) -> Any:
        if key in self.omit_delegate_attrs:
            raise AttributeError(
                "'{}' does not delegate attribute {}".format(
                    self.__class__.__name__, key
                )
            )

        for name in self.delegate_attr_dicts:
            if key == name:
                # needed to prevent infinite loops!
                raise AttributeError(
                    "dict '{}' has not been created in object '{}'".format(
                        key, self.__class__.__name__
                    )
                )
            try:
                d = getattr(self, name, None)
                if d is not None:
                    return d[key]
            except KeyError:
                pass

        for name in self.delegate_attr_objects:
            if key == name:
                raise AttributeError(
                    "object '{}' has not been created in object '{}'".format(
                        key, self.__class__.__name__
                    )
                )
            try:
                obj = getattr(self, name, None)
                if obj is not None:
                    return getattr(obj, key)
            except AttributeError:
                pass

        raise AttributeError(
            "'{}' object and its delegates have no attribute '{}'".format(
                self.__class__.__name__, key
            )
        )

    def __dir__(self) -> List[str]:
        names = list(super().__dir__())
        for name in self.delegate_attr_dicts:
            d = getattr(self, name, None)
            if d is not None:
                names += [k for k in d.keys() if k not in self.omit_delegate_attrs]

        for name in self.delegate_attr_objects:
            obj = getattr(self, name, None)
            if obj is not None:
                names += [k for k in dir(obj) if k not in self.omit_delegate_attrs]

        return sorted(set(names))


def strip_attrs(obj: object, whitelist: Sequence[str] = ()) -> None:
    """
    Irreversibly remove all direct instance attributes of object, to help with
    disposal, breaking circular references.

    Args:
        obj: Object to be stripped.
        whitelist: List of names that are not stripped from the object.
    """
    try:
        lst = set(list(obj.__dict__.keys())) - set(whitelist)
        for key in lst:
            try:
                del obj.__dict__[key]
            # TODO (giulioungaretti) fix bare-except
            except:
                pass
        # TODO (giulioungaretti) fix bare-except
    except:
        pass


def checked_getattr(
    instance: Any, attribute: str, expected_type: Union[type, Tuple[type, ...]]
) -> Any:
    """
    Like ``getattr`` but raises type error if not of expected type.
    """
    attr: Any = getattr(instance, attribute)
    if not isinstance(attr, expected_type):
        raise TypeError()
    return attr


@contextmanager
def attribute_set_to(
    object_: object, attribute_name: str, new_value: Any
) -> Iterator[None]:
    """
    This context manager allows to change a given attribute of a given object
    to a new value, and the original value is reverted upon exit of the context
    manager.

    Args:
        object_: The object which attribute value is to be changed.
        attribute_name: The name of the attribute that is to be changed.
        new_value: The new value to which the attribute of the object is
                   to be changed.
    """
    old_value = getattr(object_, attribute_name)
    setattr(object_, attribute_name, new_value)
    try:
        yield
    finally:
        setattr(object_, attribute_name, old_value)
