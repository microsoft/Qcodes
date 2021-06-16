"""
Author: Victor Negirneac <vnegirneac@qblox.com>

A sphinx extension that patches several parts of sphinx and autodoc in order
to allow nice auto-documentation of parameters added with
`@qcodes.instrument.base.add_parameter`.

The implementation is hacky but attempts to be minimally intrusive and somewhat
resilient to future changes in patched methods.

The qcodes parameters of an Instrument added with the
`@qcodes.instrument.base.add_parameter`
can be referenced in the .rst files with the ``:obj:`` or ``:meth:`` roles.
"""
# referencing with :parameter: would be nicer but seemed challenging to implement
import inspect
from sphinx.ext import autodoc
from sphinx.domains import python as sphinx_domains_python

from qcodes.instrument.base import ADD_PARAMETER_ATTR_NAME, DECORATED_METHOD_PREFIX

# ######################################################################################
# Rename the special method and prefix it with "parameter " in docs output
# ######################################################################################

def format_name(self) -> str:
    if hasattr(self.object, ADD_PARAMETER_ATTR_NAME):
        return ".".join(
            self.objpath[:-1] + [self.objpath[-1][len(DECORATED_METHOD_PREFIX) :]]
        )
    return ".".join(self.objpath) or self.modname


original_add_directive_header = autodoc.MethodDocumenter.add_directive_header


def add_directive_header(self, sig: str) -> None:
    original_add_directive_header(self, sig)
    if self.object_name.startswith("_parameter_"):
        sourcename = self.get_sourcename()
        self.add_line("   :parameter:", sourcename)


original_get_signature_prefix = sphinx_domains_python.PyMethod.get_signature_prefix


def get_signature_prefix(self, sig: str) -> str:
    prefix_str = original_get_signature_prefix(self, sig)
    if "parameter" in self.options:
        prefix_str += "parameter "

    return prefix_str


# ######################################################################################
# Move parameter instantiation specification to the docstring and remove from signature
# ######################################################################################

def add_parameter_spec_to_docstring(app, what, name, obj, options, lines):
    # name is e.g. `"my_module._parameter_time"
    modify_dosctring = app.config["qcodes_parameters_spec_in_docstring"]
    add_links = "~" if app.config["qcodes_parameters_spec_with_links"] else "!"
    if modify_dosctring and DECORATED_METHOD_PREFIX in name:
        lines += [
            "",
            ".. rubric:: Arguments passed to "
            f":meth:`{add_links}qcodes.instrument.base.InstrumentBase.add_parameter`:",
            "",
        ]
        for kw_name, par_obj in inspect.signature(obj).parameters.items():
            if kw_name != "self":
                if kw_name == "parameter_class":
                    # create link to the parameter class
                    mod = par_obj.default.__module__
                    class_name = par_obj.default.__name__
                    value = f":class:`{add_links}{mod}.{class_name}`"
                else:
                    value = f"*{par_obj.default!r}*"

                if add_links == "~" and kw_name == "vals":
                    kw_name = ":mod:`vals <qcodes.utils.validators>`"
                else:
                    kw_name = f"**{kw_name}**"
                lines.append(f"- {kw_name} = {value}")
        lines += [""]

    return lines


def clear_parameter_method_signature(
    app, what, name, obj, options, signature, return_annotation
):
    modify_dosctring = app.config["qcodes_parameters_spec_in_docstring"]
    if modify_dosctring and DECORATED_METHOD_PREFIX in name:
        signature = "()"
    return (signature, return_annotation)

# ######################################################################################
# Setup sphinx extension
# ######################################################################################

def dont_skip_qcodes_params(app, what, name, obj, skip, options) -> bool:
    if hasattr(obj, ADD_PARAMETER_ATTR_NAME):
        return False
    return skip

def setup(app):
    app.setup_extension('sphinx.ext.autodoc')  # Require autodoc extension
    app.add_config_value(
        name="qcodes_parameters_spec_in_docstring",
        default=True,
        rebuild="html",
        types=(bool)
    )
    app.add_config_value(
        name="qcodes_parameters_spec_with_links",
        default=True,
        rebuild="html",
        types=(bool)
    )
    # we can't have a method in the class with the same name as the desired
    # parameter, therefore we patch the method name displayed in the sphinx docs
    # monkey patching `MethodDocumenter`
    # e.g. `_parameter_time` -> `_parameter_time` when displayed in the docs
    autodoc.MethodDocumenter.format_name = format_name
    autodoc.MethodDocumenter.add_directive_header = add_directive_header
    sphinx_domains_python.PyMethod.option_spec.update(
        {"parameter": sphinx_domains_python.PyMethod.option_spec["classmethod"]}
    )
    sphinx_domains_python.PyMethod.get_signature_prefix = get_signature_prefix
    # enforce documenting the decorated private method
    app.connect("autodoc-skip-member", dont_skip_qcodes_params)
    app.connect("autodoc-process-docstring", add_parameter_spec_to_docstring)
    app.connect("autodoc-process-signature", clear_parameter_method_signature)

    return {
        'version': '0.1',
        'parallel_read_safe': True,  # Not tested, should not be an issue
        'parallel_write_safe': True,  # Not tested, should not be an issue
    }
