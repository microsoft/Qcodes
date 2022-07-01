from qcodes.extensions.sphinx_extension.parse_parameter_attr import (
    ParameterProxy,
    eval_params_from_code,
    extract_code_as_repr,
    extract_statements_from_node,
    find_class,
    find_init_func,
    parse_init_function_from_str,
    qcodes_parameter_attr_getter,
    setup,
)
from qcodes.utils import issue_deprecation_warning

issue_deprecation_warning(
    "QCoDeS Sphinx extension: qcodes_parameter_attr_getter has been removed "
    "from the public API. Please raise an issue at github.com/qcodes/qcodes "
    "if you rely on this outside qcodes"
)
