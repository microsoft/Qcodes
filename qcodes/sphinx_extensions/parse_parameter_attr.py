import inspect
from typing import Any, Dict, Optional, Tuple, Type, Union

import parso
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


def parse_init_function_from_str(
    code: str, classname: str
) -> Optional[parso.python.tree.Function]:
    module = parso.parse(code)
    classes = tuple(
        child
        for child in module.children
        if isinstance(child, parso.python.tree.Class) and child.name.value == classname
    )
    if len(classes) != 1:

        LOGGER.warning(f"Could not find exactly one class for {classname}")
        return None
    assert len(classes) == 1
    myclass = classes[0]
    nodes = tuple(
        child
        for child in myclass.children
        if isinstance(child, parso.python.tree.PythonNode)
    )
    node = nodes[-1]
    init_funcs = tuple(
        child
        for child in node.children
        if isinstance(child, parso.python.tree.Function)
        and child.name.value == "__init__"
    )
    if len(init_funcs) != 1:
        LOGGER.warning(
            f"Did not find an init func or found more than one from {init_funcs}"
        )
        return None
    return init_funcs[0]


def extract_statements_from_func_node(
    parso_func: parso.python.tree.Function,
) -> Tuple[parso.python.tree.ExprStmt, ...]:
    function_bodys = tuple(
        child
        for child in parso_func.children
        if isinstance(child, parso.python.tree.PythonNode) and child.type == "suite"
    )
    assert len(function_bodys) == 1
    function_body = function_bodys[0]
    statement_lines = tuple(
        child.children[0]
        for child in function_body.children
        if isinstance(child, parso.python.tree.PythonNode)
        and isinstance(child.children[0], parso.python.tree.ExprStmt)
    )

    return statement_lines


def eval_params_from_code(code: str, classname: str) -> Dict[str, ParameterProxy]:
    init_func_tree = parse_init_function_from_str(code, classname)
    if init_func_tree is None:
        return {}
    stms = extract_statements_from_func_node(init_func_tree)
    param_dict = {}

    for stm in stms:
        try:
            name_code = extract_code_as_repr(stm)
        except Exception:
            continue
        if name_code is not None:
            name, proxy_param = name_code
            param_dict[name] = proxy_param
    return param_dict


def extract_code_as_repr(
    stm: parso.python.tree.ExprStmt,
) -> Optional[Tuple[str, ParameterProxy]]:
    lhs = stm.children[0]
    rhs = stm.get_rhs()

    if isinstance(lhs, parso.python.tree.BaseNode) and len(lhs.children) == 2:
        obj1 = lhs.children[0]
        obj2 = lhs.children[1]
        if (
            isinstance(obj1, parso.python.tree.Leaf)
            and obj1.value == "self"
            and isinstance(obj2, parso.python.tree.BaseNode)
            and isinstance(obj2.children[1], parso.python.tree.Leaf)
        ):
            name = obj2.children[1].value
            code = " ".join(rhs.get_code().strip().split())
            pp = ParameterProxy(code)
            return name, pp
        else:
            return None
    else:
        return None


def qcodes_parameter_attr_getter(
    object_to_document_attr_on: Type[object], name: str, *default: Any
) -> Any:
    if (
        inspect.isclass(object_to_document_attr_on)
        and issubclass(object_to_document_attr_on, InstrumentBase)
        and not name.startswith("_")
    ):
        try:
            attr = safe_getattr(object_to_document_attr_on, name)
        except AttributeError as e:
            LOGGER.debug(f"Parsing attribute {name} on {object_to_document_attr_on}")
            obj_name = object_to_document_attr_on.__name__
            with open(
                inspect.getfile(object_to_document_attr_on), encoding="utf8"
            ) as file:
                code = file.read()
            param_dict = eval_params_from_code(code, obj_name)
            if param_dict.get(name) is not None:
                attr = param_dict[name]
            else:
                LOGGER.debug(
                    f"fall back to default for {name} on {object_to_document_attr_on}"
                )
                attr = safe_getattr(object_to_document_attr_on, name, default)
    else:
        attr = safe_getattr(object_to_document_attr_on, name, default)
    return attr


def setup(app: Any) -> Dict[str, Union[str, bool]]:
    """Called by sphinx to setup the extension."""
    app.setup_extension("sphinx.ext.autodoc")  # Require autodoc extension

    app.add_autodoc_attrgetter(object, qcodes_parameter_attr_getter)

    return {
        "version": "0.1",
        "parallel_read_safe": True,  # Not tested, should not be an issue
        "parallel_write_safe": True,  # Not tested, should not be an issue
    }
