"""A Sphinx extension that uses Parso to extract the code of a instance attribute.

The code is used to produce a proxy object with a repr containing the code defining the
attribute. This enables better documentation of instance attributes.
Especially QCoDeS Parameters. Note that this is for the moment limited to
attributes on QCoDeS instruments."""

import functools
import inspect
from typing import Any

import parso
import parso.python
import parso.python.tree
import parso.tree
from sphinx.util import logging
from sphinx.util.inspect import safe_getattr

from qcodes.instrument.base import InstrumentBase

LOGGER = logging.getLogger(__name__)


class ParameterProxy:
    """
    An object that acts as a proxy for documenting containing
    a repr that can be set from a string.

    """

    def __init__(self, repr_str: str):
        self._repr = repr_str

    def __repr__(self) -> str:
        """
        A repr based on the string set in init.
        """
        return self._repr


def find_class(
    node: parso.tree.BaseNode, classname: str
) -> tuple[parso.python.tree.Class, ...]:
    """Find all classes in a given Parso node named ``classname``."""
    nodes = []
    for child in node.children:
        if (
            isinstance(child, parso.python.tree.Class) and child.name.value == classname  # pyright: ignore
        ):
            nodes.append(child)
        elif isinstance(child, parso.tree.Node):
            nodes.extend(find_class(child, classname))
    return tuple(nodes)


def find_init_func(
    node: parso.tree.BaseNode,
) -> tuple[parso.python.tree.Function, ...]:
    """Find all ``__init__`` functions in the supplied Parso node."""
    nodes = []
    for child in node.children:
        if (
            isinstance(child, parso.python.tree.Function)
            and child.name.value  # pyright: ignore[reportAttributeAccessIssue]
            == "__init__"
        ):
            nodes.append(child)
        elif isinstance(child, parso.tree.Node):
            nodes.extend(find_init_func(child))
    return tuple(nodes)


def parse_init_function_from_str(
    code: str, classname: str
) -> parso.python.tree.Function | None:
    module = parso.parse(code)
    classes = find_class(module, classname)
    if len(classes) > 1:
        LOGGER.warning(
            f"Found more than one class definition for {classname}: Found {classes}"
        )
        return None
    if len(classes) == 0:
        LOGGER.debug(f"Could not find a class definition for {classname}")
        return None
    init_funcs = find_init_func(classes[0])
    if len(init_funcs) > 1:
        LOGGER.warning(
            f"Found more than one init function for {classname}: Found {init_funcs}"
        )
        return None
    if len(init_funcs) == 0:
        LOGGER.debug(f"Found no init function for {classname}")
        return None
    return init_funcs[0]


def extract_statements_from_node(
    parso_node: parso.tree.BaseNode,
) -> tuple[parso.python.tree.ExprStmt, ...]:
    nodes = []
    for child in parso_node.children:
        if isinstance(child, parso.python.tree.ExprStmt):
            nodes.append(child)
        elif isinstance(child, parso.tree.Node):
            nodes.extend(extract_statements_from_node(child))
    return tuple(nodes)


@functools.lru_cache(maxsize=None, typed=True)
def eval_params_from_code(code: str, classname: str) -> dict[str, ParameterProxy]:
    init_func_tree = parse_init_function_from_str(code, classname)
    if init_func_tree is None:
        return {}
    stms = extract_statements_from_node(init_func_tree)
    param_dict = {}

    for stm in stms:
        try:
            name_code = extract_code_as_repr(stm)
        except Exception:
            LOGGER.warning(f"Error while trying to parse attribute from {classname}")
            continue
        if name_code is not None:
            name, proxy_param = name_code
            param_dict[name] = proxy_param
    return param_dict


def extract_code_as_repr(
    stm: parso.python.tree.ExprStmt,
) -> tuple[str, ParameterProxy] | None:
    lhs = stm.children[0]
    rhs = stm.get_rhs()

    if isinstance(lhs, parso.tree.BaseNode) and len(lhs.children) == 2:
        obj1 = lhs.children[0]
        obj2 = lhs.children[1]
        if (
            isinstance(obj1, parso.tree.Leaf)
            and obj1.value == "self"
            and isinstance(obj2, parso.tree.BaseNode)
            and isinstance(obj2.children[1], parso.tree.Leaf)
        ):
            name = obj2.children[1].value
            code_str = rhs.get_code()
            assert code_str is not None
            code = " ".join(code_str.strip().split())
            pp = ParameterProxy(code)
            return name, pp
        else:
            return None
    else:
        return None


def qcodes_parameter_attr_getter(
    object_to_document_attr_on: type[object], name: str, *default: Any
) -> Any:
    """
    Try to extract an attribute as a proxy object with a repr containing the code
    if the class the attribute is bound to is a subclass of ``InstrumentBase``
    and the attribute is not private.

    Args:
        object_to_document_attr_on: The type (not instance of the object to detect the
            attribute on.
        name: Name of the attribute to look for.
        *default: Default obejct to use as a replacement if the attribute could not be
            found.

    Returns:
        Attribute looked up, proxy object containing the code of the attribute as a
        repr or a default object.
    """
    if (
        inspect.isclass(object_to_document_attr_on)
        and issubclass(object_to_document_attr_on, InstrumentBase)
        and not name.startswith("_")
    ):
        try:
            attr = safe_getattr(object_to_document_attr_on, name)
        except AttributeError:
            LOGGER.debug(
                f"Attempting to load attribute {name} on "
                f"{object_to_document_attr_on} via parsing"
            )
            mro = inspect.getmro(object_to_document_attr_on)
            attr = None
            for classobj in mro:
                try:
                    param_dict = eval_params_from_code(
                        inspect.getsource(classobj), classobj.__name__
                    )
                    if param_dict.get(name) is not None:
                        attr = param_dict[name]
                        break
                except TypeError:
                    continue
            if attr is None:
                LOGGER.debug(
                    f"Falling back to default Sphinx attribute loader for {name}"
                    f" on {object_to_document_attr_on}"
                )
                attr = safe_getattr(object_to_document_attr_on, name, *default)
    else:
        attr = safe_getattr(object_to_document_attr_on, name, *default)
    return attr


def setup(app: Any) -> dict[str, str | bool]:
    """Called by sphinx to setup the extension."""
    app.setup_extension("sphinx.ext.autodoc")  # Require autodoc extension

    app.add_autodoc_attrgetter(object, qcodes_parameter_attr_getter)

    return {
        "version": "0.1",
        "parallel_read_safe": True,  # Not tested, should not be an issue
        "parallel_write_safe": True,  # Not tested, should not be an issue
    }
