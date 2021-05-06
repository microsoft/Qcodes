""" Representation methods for instruments """
import logging
import uuid
from collections import namedtuple
from functools import partial
from html import escape

import numpy as np
from xarray.core.formatting_html import _icon, _mapping_section, _obj_repr


def create_entry(ed, preview=None):
    name = ed.name
    preview = ed.short
    long = ed.long

    data_id = "data-" + str(uuid.uuid4())
    data_icon = _icon("icon-database")
    data_repr = f"{long}"
    cssclass_idx = " class='xr-has-index'"

    logging.debug(f"create_entry: {name} {preview}")

    disabled = "" if long is not None else "disabled"

    return (
        f"<div class='xr-var-name'><span{cssclass_idx}>{name}</span></div>"
        f"<div class='xr-var-preview xr-preview'>{preview}</div>"
        f"<input id='{data_id}' class='xr-var-data-in' type='checkbox' {disabled}>"
        f"<label for='{data_id}' title='Show/Hide data repr'>"
        f"{data_icon}</label>"
        f"<div class='xr-var-data'>{data_repr}</div>"
    )


def create_entries(variables):
    logging.debug(f"create_entries: {variables}")
    vars_li = "".join(
        f"<li class='xr-var-item'>{create_entry(entry)}</li>"
        for name, entry in variables.items()
    )

    return f"<ul class='xr-var-list'>{vars_li}</ul>"


parameter_section = partial(
    _mapping_section,
    name="Parameters",
    details_func=create_entries,
    max_items_collapse=10,
)

functions_section = partial(
    _mapping_section,
    name="Functions",
    details_func=create_entries,
    max_items_collapse=10,
)


def instrument_repr_html(self):
    obj_type = "{}".format(type(self).__name__)

    name = self.name
    header_components = [f"<div class='xr-obj-type'>{escape(obj_type)}: {name}</div>"]

    Entry = namedtuple("SectionEntry", ["name", "short", "long"])

    dd = []
    parameters = list(self.parameters.keys())
    for ii, parameter_name in enumerate(sorted(parameters)):
        parameter = self.parameters[parameter_name]
        try:
            short = f"{parameter()} [{parameter.unit}]"
        except:
            short = None
        timestamp = parameter.cache.timestamp

        def hcolor(txt):
            return f'<span style="color:#777777">{txt}</span>'

        long = f"unit: {hcolor(parameter.unit)}, label: {hcolor(parameter.label)}, timestamp: {hcolor(timestamp)}"
        dd.append(Entry(parameter_name, short=short, long=long))
    section_data = {ed.name: ed for ed in dd}

    functions_section_data = {
        key: Entry(key, str(function), None) for key, function in self.functions.items()
    }

    sections = [
        parameter_section(section_data),
        functions_section(functions_section_data),
    ]

    return _obj_repr(self, header_components, sections)
