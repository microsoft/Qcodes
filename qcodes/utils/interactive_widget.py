"""This file contains functions to displays an interactive widget
with information about `qcodes.experiments()`."""

import math
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import qcodes
import yaml
from IPython.core.display import display
from IPython.display import clear_output
from ipywidgets import (
    HTML,
    Box,
    Button,
    GridspecLayout,
    Label,
    Layout,
    Output,
    Tab,
    Textarea,
    VBox,
)
from qcodes.dataset import initialise_or_create_database_at
from qcodes.dataset.data_export import get_data_by_id
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.plotting import plot_dataset
from toolz.dicttoolz import get_in

from formatting_html import _repr_html_

_META_DATA_KEY = "widget_notes"


def button(
    description: str,
    button_style: Optional[str] = None,
    on_click: Optional[Callable[[Any], None]] = None,
    tooltip: Optional[str] = None,
    layout_kwargs: Optional[Dict[str, Any]] = None,
    button_kwargs: Optional[Dict[str, Any]] = None,
) -> Button:
    """Returns a ipywidgets.Button."""
    layout_kwargs = layout_kwargs or {}
    but = Button(
        description=description,
        button_style=button_style,
        layout=Layout(
            height=layout_kwargs.pop("height", "auto"),
            width=layout_kwargs.pop("width", "auto"),
            **layout_kwargs,
        ),
        tooltip=tooltip or description,
        **(button_kwargs or {}),
    )
    if on_click is not None:
        but.on_click(on_click)
    return but


def text(description: str) -> Label:
    """Returns a `ipywidgets.Label` with text."""
    return Label(value=description, layout=Layout(height="max-content", width="auto"))


def _update_nested_dict_browser(
    nested_keys: Sequence[str], table: Dict[Any, Any], box: Box
) -> Callable[[Button], None]:
    def _(_):
        box.children = (_nested_dict_browser(nested_keys, table, box),)

    return _


def _nested_dict_browser(
    nested_keys: Sequence[str], table: Dict[Any, Any], box: Box, max_nrows: int = 30
) -> GridspecLayout:
    """Generates a `GridspecLayout` of the ``nested_keys`` in ``table`` which is
    put inside of ``box``.

    Whenever the table has less than ``max_nrows`` rows, the table is
    displayed in 3 columns, otherwise it's 2 columns.
    """

    def _should_expand(x):
        return isinstance(x, dict) and x != {}

    col_widths = [8, 16, 30]
    selected_table = get_in(nested_keys, table)
    nrows = sum(len(v) if _should_expand(v) else 1 for v in selected_table.values()) + 1
    ncols = 3

    if nrows > max_nrows:
        nrows = len(selected_table) + 1
        col_widths.pop(1)
        ncols = 2

    grid = GridspecLayout(nrows, col_widths[-1])
    update = partial(_update_nested_dict_browser, table=table, box=box)

    # Header
    title = " ► ".join(nested_keys)
    grid[0, :-1] = button(title, "success")
    up_click = update(nested_keys[:-1])
    grid[0, -1] = button("↰", "info", up_click)

    # Body

    i = 1
    for k, v in selected_table.items():
        row_length = len(v) if _should_expand(v) and ncols == 3 else 1
        but = button(k, "info", up_click)
        grid[i : i + row_length, : col_widths[0]] = but
        if _should_expand(v):
            if ncols == 3:
                for k_, v_ in v.items():
                    but = button(k_, "danger", update([*nested_keys, k]))
                    grid[i, col_widths[0] : col_widths[1]] = but
                    if _should_expand(v_):
                        sub_keys = ", ".join(v_.keys())
                        but = button(sub_keys, "warning", update([*nested_keys, k, k_]))
                    else:
                        but = text(str(v_))
                    grid[i, col_widths[1] :] = but
                    i += 1
            else:
                sub_keys = ", ".join(v.keys())
                grid[i, col_widths[0] :] = button(
                    sub_keys, "danger", update([*nested_keys, k])
                )
                i += 1
        else:
            grid[i, col_widths[0] :] = text(str(v))
            i += 1
    return grid


def nested_dict_browser(
    nested_dict: Dict[Any, Any], nested_keys: Sequence[str] = ()
) -> Box:
    """Returns a widget to interactive browse a nested dictionary."""
    box = Box([])
    _update_nested_dict_browser(nested_keys, nested_dict, box)(None)
    return box


def _plot_ds(ds: DataSet) -> None:
    try:
        # `get_data_by_id` might fail
        nplots = len(get_data_by_id(ds.run_id))  # TODO: might be a better way
        nrows = math.ceil(nplots / 2) if nplots != 1 else 1
        ncols = 2 if nplots != 1 else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        # `plot_dataset` might also fail.
        plot_dataset(ds, axes=axes.flatten())
        fig.tight_layout()
        plt.show(fig)
    except Exception as e:
        print(e)  # TODO: print complete traceback


def _do_in_tab(tab: Tab, ds: DataSet, which: str) -> Callable[[Button], None]:
    """Performs an operation inside of a subtab of a `ipywidgets.Tab`.

    Args
        tab: Instance of `ipywidgets.Tab`
        ds: A DataSet
        which: can be either "plot", "snapshot", or "dataset"
    """

    def delete_tab(output, tab):
        def on_click(_):
            tab.children = tuple(c for c in tab.children if c != output)

        return on_click

    def _on_click(_):
        assert which in ("plot", "snapshot", "dataset")
        title = f"RID #{ds.run_id} {which}"
        i = next(
            (i for i in range(len(tab.children)) if tab.get_title(i) == title), None
        )
        if i is not None:
            # Plot/snapshot is already in the tab
            tab.selected_index = i
            return
        out = Output()
        tab.children += (out,)
        i = len(tab.children) - 1
        tab.set_title(i, title)
        with out:
            clear_output(wait=True)
            try:
                if which == "plot":
                    _plot_ds(ds)
                elif which == "snapshot":
                    display(nested_dict_browser(ds.snapshot))
                elif which == "dataset":
                    display(_repr_html_(ds))
            except Exception as e:
                print(e)  # TODO: print complete traceback

            remove_button = button(
                f"Clear {which}",
                "danger",
                on_click=delete_tab(out, tab),
                button_kwargs=dict(icon="eraser"),
            )
            display(remove_button)
        tab.selected_index = i

    return _on_click


def create_tab(do_display: bool = True) -> Tab:
    """Creates a `ipywidgets.Tab` which can display outputs in its tabs."""
    tab = Tab(children=(Output(),))

    tab.set_title(0, "Info")
    if do_display:
        display(tab)

    with tab.children[-1]:
        print("Plots and snapshots will show up here!")
    return tab


def editable_metadata(ds: DataSet) -> Box:
    def _button_to_input(text, box):
        def on_click(_):
            text_input = Textarea(
                value=text,
                placeholder="Enter text",
                disabled=False,
                layout=Layout(height="auto", width="auto"),
            )
            save_button = button(
                "",
                "danger",
                on_click=_save_button(box, ds),
                button_kwargs=dict(icon="save"),
            )
            box.children = (text_input, save_button)

        return on_click

    def _save_button(box, ds):
        def on_click(_):
            text = box.children[0].value
            ds.add_metadata(tag=_META_DATA_KEY, metadata=text)
            box.children = (_changeable_button(text, box),)

        return on_click

    def _changeable_button(text, box):
        return button(
            text,
            "success",
            on_click=_button_to_input(text, box),
            button_kwargs=dict(icon="edit") if text == "" else {},
        )

    text = ds.metadata.get(_META_DATA_KEY, "")
    box = VBox([], layout=Layout(height="auto", width="auto"))
    box.children = (_changeable_button(text, box),)
    return box


def expandable_dict(dct, tab, ds):
    """Returns a `ipywidgets.Button` which on click changes into a text area
    and buttons, that when clicked show something in a subtab of ``tab``."""

    def _button_to_input(dct, box):
        def on_click(_):
            description = yaml.dump(dct)  # TODO: include and extract more data!
            text_input = Textarea(
                value=description,
                placeholder="Enter text",
                disabled=True,
                layout=Layout(height="300px", width="auto"),
            )
            plot_button = button(
                "Plot",
                "warning",
                on_click=_do_in_tab(tab, ds, "plot"),
                button_kwargs=dict(icon="line-chart"),
            )
            snapshot_button = button(
                "Open snapshot",
                "warning",
                on_click=_do_in_tab(tab, ds, "snapshot"),
                button_kwargs=dict(icon="camera"),
            )
            dataset_button = button(
                "Inpect dataset",
                "warning",
                on_click=_do_in_tab(tab, ds, "dataset"),
                button_kwargs=dict(icon="search"),
            )
            back_button = button(
                "Back",
                "warning",
                on_click=_input_to_button(dct, box),
                button_kwargs=dict(icon="undo"),
            )
            box.children = (
                text_input,
                snapshot_button,
                dataset_button,
                plot_button,
                back_button,
            )

        return on_click

    def _input_to_button(dct, box):
        def on_click(_):
            box.children = (_changeable_button(dct, box),)

        return on_click

    def _changeable_button(dct, box):
        return button(
            ", ".join(dct),
            "success",
            on_click=_button_to_input(dct, box),
            button_kwargs=dict(icon="edit") if text == "" else {},
        )

    box = VBox([], layout=Layout(height="auto", width="auto"))
    box.children = (_changeable_button(dct, box),)
    return box


def _get_coords_and_vars(ds: DataSet) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    coordinates = {}
    variables = {}
    for p, spec in ds.paramspecs.items():
        attrs = {
            "unit": spec.unit,
            "label": spec.label,
            "type": spec.type,
        }
        if spec.depends_on:
            attrs["depends_on"] = spec.depends_on.split(", ")
            variables[p] = attrs
        else:
            coordinates[p] = attrs
    return coordinates, variables


def _experiment_widget(tab: Tab) -> GridspecLayout:
    """Show a `ipywidgets.GridspecLayout` with information about the
    loaded experiment. The clickable buttons can perform an action in ``tab``.
    """
    header_names = [
        "Run ID",
        "Name",
        "Experiment",
        "Coordinates",
        "Variables",
        "MSMT Time",
        "Notes",
    ]

    header = {n: button(n, "info") for n in header_names}
    rows = [header]
    for exp in qcodes.experiments():
        tooltip = f"{exp.name}#{exp.sample_name}@{exp.path_to_db}"

        for ds in exp.data_sets():
            coords, variables = _get_coords_and_vars(ds)
            row = {}
            row["Run ID"] = text(str(ds.run_id))
            row["Name"] = text(ds.name)
            row["Experiment"] = button(f"#{exp.exp_id}", "warning", tooltip=tooltip)
            row["Notes"] = editable_metadata(ds)
            row["Coordinates"] = expandable_dict(coords, tab, ds)
            row["Variables"] = expandable_dict(variables, tab, ds)
            row["MSMT Time"] = text(ds.completed_timestamp() or "")
            rows.append(row)

    grid = GridspecLayout(n_rows=len(rows), n_columns=len(header_names))

    empty_text = text("")
    for i, row in enumerate(rows):
        for j, name in enumerate(header_names):
            grid[i, j] = row.get(name, empty_text)

    grid.layout.grid_template_rows = "auto " * len(rows)
    grid.layout.grid_template_columns = "auto " * len(header_names)
    return grid


def experiments_widget(db: Optional[str] = None) -> VBox:
    f"""Displays an interactive widget that shows the ``qcodes.experiments()``.

    Using the edit button in the column "Notes", one can make persistent changes
    to the `~qcodes.dataset.data_set.DataSet`\s attribute ``metadata``
    in the key "{_META_DATA_KEY}".
    Expanding the coordinates or variables buttons, reveals more options, such as
    plotting or the ability to easily browse
    the `~qcodes.dataset.data_set.DataSet`\s snapshot.

    Args
        db: Optionally pass a database file, if no database has been loaded.
    """
    if db is not None:
        initialise_or_create_database_at(db)
    title = HTML("<h1>QCoDeS experiments widget</h1>")
    tab = create_tab(do_display=False)
    grid = _experiment_widget(tab)
    return VBox([title, tab, grid])
