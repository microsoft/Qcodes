import inspect
from typing import Any, Dict, Optional, Tuple

import numpy as np
import parso
from sphinx.util.inspect import safe_getattr

# adhock imports required to run eval below
# this should be parsed out rather
import qcodes.utils.validators as vals
from qcodes.instrument.base import InstrumentBase
from qcodes.instrument.parameter import Parameter


def parse_init_function_from_str(
    code: str, classname
) -> Optional[parso.python.tree.Function]:
    module = parso.parse(code)
    classes = tuple(
        child
        for child in module.children
        if isinstance(child, parso.python.tree.Class) and child.name.value == classname
    )
    if len(classes) != 1:
        print(f"Could not find exactly one class for {classname}")
        return None
    assert len(classes) == 1
    myclass = classes[0]
    nodes = tuple(
        child
        for child in myclass.children
        if isinstance(child, parso.python.tree.PythonNode)
    )
    if len(nodes) != 1:
        print(f"Could not find a single node from {classname}")
        return None
    node = nodes[0]
    init_funcs = tuple(
        child
        for child in node.children
        if isinstance(child, parso.python.tree.Function)
        and child.name.value == "__init__"
    )
    if len(init_funcs) != 1:
        print(f"Did not find an init func or found more than one from {init_funcs}")
        return None
    return init_funcs[0]


def extract_statements_from_func_node(parso_func: parso.python.tree.Function):
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


def eval_params_from_code(code: str, classname: str) -> Dict[str, Parameter]:
    init_func_tree = parse_init_function_from_str(code, classname)
    if init_func_tree is None:
        return {}
    stms = extract_statements_from_func_node(init_func_tree)
    param_dict = {}

    for stm in stms:
        try:
            name_code = extract_code_without_self_from_statement(stm)
        except:
            continue
        if name_code is not None:
            name, code = name_code
            try:
                param_dict[name] = eval(code)
            except Exception as e:
                param_dict[name] = None
    return param_dict


def parse_string_or_node(stm):
    skip = False
    if isinstance(stm, parso.python.tree.PythonNode):
        for child in stm.children:
            if not isinstance(child, parso.python.tree.PythonNode):
                # todo more robust parsing of recursive nodes
                if child.value == "self":
                    skip = True
            if isinstance(child, parso.python.tree.PythonNode):
                maybeskip = parse_string_or_node(child)
                if maybeskip:
                    skip = True
    else:
        if stm.value == "self":
            skip = True

    return skip


def extract_code_without_self_from_statement(
    stm: parso.python.tree.ExprStmt,
) -> Optional[Tuple[str, str]]:
    lhs = stm.children[0]
    rhs = stm.get_rhs()
    if len(lhs.children) == 2 and lhs.children[0].value == "self":
        name = lhs.children[1].children[1].value
        arglist = rhs.children[1].children[1]
        to_remove = []
        for i, arg in enumerate(arglist.children):
            if parse_string_or_node(arg):
                to_remove.append(i)
                to_remove.append(i + 1)
        to_remove.sort(reverse=True)
        for j in to_remove:
            arglist.children.pop(j)
        return name, rhs.get_code().strip()
    else:
        return None


def qcodes_parameter_attr_getter(object: Any, name: str, *default: Any) -> Any:
    if (
        inspect.isclass(object)
        and issubclass(object, InstrumentBase)
        and not name.startswith("_")
    ):
        try:
            return safe_getattr(object, name)
        except AttributeError:
            print(f"Parsing attribute {name} on {object}")
            obj_name = object.__name__
            with open(inspect.getfile(object), encoding="utf8") as file:
                code = file.read()
            param_dict = eval_params_from_code(code, obj_name)
            if param_dict.get(name) is not None:
                return param_dict[name]
            else:
                print("fall back to default")
                return safe_getattr(object, name, default)
    else:

        return safe_getattr(object, name, default)


def setup(app):
    """Called by sphinx to setup the extension."""
    app.setup_extension("sphinx.ext.autodoc")  # Require autodoc extension

    app.add_autodoc_attrgetter(object, qcodes_parameter_attr_getter)

    return {
        "version": "0.1",
        "parallel_read_safe": True,  # Not tested, should not be an issue
        "parallel_write_safe": True,  # Not tested, should not be an issue
    }
