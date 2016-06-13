"""Nested attribute / item access for use by remote proxies."""

import re


class _NoDefault:

    """Empty class to provide a missing default to getattr."""


class NestedAttrAccess:

    """
    A Mixin class to provide nested access to attributes and their items.

    Primarily for use by remote proxies, so we don't need to separately
    proxy all the components, and all of their components, and worry about
    which are picklable, etc.
    """

    def getattr(self, attr, default=_NoDefault):
        """
        Get a (possibly nested) attribute of this object.

        If there is no ``.`` or ``[]`` in ``attr``, this exactly matches
        the ``getattr`` function, but you can also access smaller pieces.

        Args:
            attr (str): An attribute or accessor string, like:
                ``'attr.subattr[item]'``. ``item`` can be an integer or a
                string. If it's a string it must be quoted.
            default (any): If the attribute does not exist (at any level of
                nesting), we return this. If no default is provided, throws
                an ``AttributeError``.

        Returns:
            The value of this attribute.

        Raises:
            ValueError: If ``attr`` could not be understood.
            AttributeError: If the attribute is missing and no default is
                provided.
            KeyError: If the item cannot be found and no default is provided.
        """
        parts = self._split_attr(attr)

        # import pdb; pdb.set_trace()

        try:
            return self._follow_parts(parts)

        except (AttributeError, KeyError):
            if default is _NoDefault:
                raise
            else:
                return default

    def setattr(self, attr, value):
        """
        Set a (possibly nested) attribute of this object.

        If there is no ``.`` or ``[]`` in ``attr``, this exactly matches
        the ``setattr`` function, but you can also access smaller pieces.

        Args:
            attr (str): An attribute or accessor string, like:
                ``'attr.subattr[item]'``. ``item`` can be an integer or a
                string; If it's a string it must be quoted as usual.

            value (any): The object to store in this attribute.

        Raises:
            ValueError: If ``attr`` could not be understood

            TypeError: If an intermediate nesting level is not a container
                and the next level is an item.

            AttributeError: If an attribute with this name cannot be set.
        """
        parts = self._split_attr(attr)
        obj = self._follow_parts(parts[:-1])
        leaf = parts[-1]

        if str(leaf).startswith('.'):
            setattr(obj, leaf[1:], value)
        else:
            obj[leaf] = value

    def delattr(self, attr):
        """
        Delete a (possibly nested) attribute of this object.

        If there is no ``.`` or ``[]`` in ``attr``, this exactly matches
        the ``delattr`` function, but you can also access smaller pieces.

        Args:
            attr (str): An attribute or accessor string, like:
                ``'attr.subattr[item]'``. ``item`` can be an integer or a
                string; If it's a string it must be quoted as usual.

        Raises:
            ValueError: If ``attr`` could not be understood
        """
        parts = self._split_attr(attr)
        obj = self._follow_parts(parts[:-1])
        leaf = parts[-1]

        if str(leaf).startswith('.'):
            delattr(obj, leaf[1:])
        else:
            del obj[leaf]

    def callattr(self, attr, *args, **kwargs):
        """
        Call a (possibly nested) method of this object.

        Args:
            attr (str): An attribute or accessor string, like:
                ``'attr.subattr[item]'``. ``item`` can be an integer or a
                string; If it's a string it must be quoted as usual.

            *args: Passed on to the method.

            **kwargs: Passed on to the method.

        Returns:
            any: Whatever the method returns.

        Raises:
            ValueError: If ``attr`` could not be understood
        """
        func = self.getattr(attr)
        return func(*args, **kwargs)

    _PARTS_RE = re.compile(r'([\.\[])')
    _ITEM_RE = re.compile(r'\[(?P<item>[^\[\]]+)\]')
    _QUOTED_RE = re.compile(r'(?P<q>[\'"])(?P<str>[^\'"]*)(?P=q)')

    def _split_attr(self, attr):
        """
        Return attr as a list of parts.

        Items in the list are:
            str '.attr' for attribute access,
            str 'item' for string dict keys,
            integers for integer dict/sequence keys.
        Other key formats are not supported
        """
        # the first item is implicitly an attribute
        parts = ('.' + self._PARTS_RE.sub(r'~\1', attr)).split('~')
        for i, part in enumerate(parts):
            item_match = self._ITEM_RE.fullmatch(part)
            if item_match:
                item = item_match.group('item')
                quoted_match = self._QUOTED_RE.fullmatch(item)
                if quoted_match:
                    parts[i] = quoted_match.group('str')
                else:
                    try:
                        parts[i] = int(item)
                    except ValueError:
                        raise ValueError('unrecognized item: ' + item)
            elif part[0] != '.' or len(part) < 2:
                raise ValueError('unrecognized attribute part: ' + part)

        return parts

    def _follow_parts(self, parts):
        obj = self

        for key in parts:
            if str(key).startswith('.'):
                obj = getattr(obj, key[1:])
            else:
                obj = obj[key]

        return obj
