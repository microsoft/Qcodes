from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


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

    delegate_attr_dicts: ClassVar[list[str]] = []
    """
    A list of names (strings) of dictionaries
    which are (or will be) attributes of ``self``, whose keys should
    be treated as attributes of ``self``.
    """
    delegate_attr_objects: ClassVar[list[str]] = []
    """
    A list of names (strings) of objects
    which are (or will be) attributes of ``self``, whose attributes
    should be passed through to ``self``.
    """
    omit_delegate_attrs: ClassVar[list[str]] = []
    """
    A list of attribute names (strings)
    to *not* delegate to any other dictionary or object.
    """

    def __getattr__(self, key: str) -> Any:
        if key in self.omit_delegate_attrs:
            raise AttributeError(
                f"'{self.__class__.__name__}' does not delegate attribute {key}"
            )

        for name in self.delegate_attr_dicts:
            if key == name:
                # needed to prevent infinite loops!
                raise AttributeError(
                    f"dict '{key}' has not been created in object '{self.__class__.__name__}'"
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
                    f"object '{key}' has not been created in object '{self.__class__.__name__}'"
                )
            try:
                obj = getattr(self, name, None)
                if obj is not None:
                    return getattr(obj, key)
            except AttributeError:
                pass

        raise AttributeError(
            f"'{self.__class__.__name__}' object and its delegates have no attribute '{key}'"
        )

    def __dir__(self) -> list[str]:
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


def strip_attrs(obj: object, whitelist: "Sequence[str]" = ()) -> None:
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
            except Exception:
                pass
    except Exception:
        pass


def checked_getattr(
    instance: Any, attribute: str, expected_type: type | tuple[type, ...]
) -> Any:
    """
    Like ``getattr`` but raises type error if not of expected type.
    """
    attr: Any = getattr(instance, attribute)
    if not isinstance(attr, expected_type):
        raise TypeError()
    return attr


def getattr_indexed(instance: Any, attribute: str) -> Any:
    """
    Similar to ``getattr`` but allows indexing the returned attribute.
    Returning a default value is _not_ supported.

    The indices are decimal digits surrounded by square brackets.
    Chained indexing is supported, but the string should not contain
    any whitespace between consecutive indices.

    Example: `getattr_indexed(some_object, "list_of_lists_field[1][2]")`
    """
    if not attribute.endswith("]"):
        return getattr(instance, attribute)

    end: int = len(attribute) - 1

    start: int = attribute.find("[", 0, end)
    attr: Any = getattr(instance, attribute[0:start])
    start += 1

    while (pos := attribute.find("][", start, end)) != -1:
        index = int(attribute[start:pos])
        attr = attr[index]
        start = pos + 2

    index = int(attribute[start:end])
    attr = attr[index]
    return attr


def checked_getattr_indexed(
    instance: Any, attribute: str, expected_type: type | tuple[type, ...]
) -> Any:
    """
    Like ``getattr_indexed`` but raises type error if not of expected type.
    """
    attr: Any = getattr_indexed(instance, attribute)
    if not isinstance(attr, expected_type):
        raise TypeError()
    return attr


@contextmanager
def attribute_set_to(
    object_: object, attribute_name: str, new_value: Any
) -> "Iterator[None]":
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
