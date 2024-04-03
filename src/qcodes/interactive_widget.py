"""This file contains functions to displays an interactive widget
with information about :func:`qcodes.dataset.experiments`."""

from __future__ import annotations

import io
import operator
import traceback
from datetime import datetime
from functools import partial, reduce
from typing import TYPE_CHECKING, Any, Literal

from ipywidgets import (  # type: ignore[import-untyped]
    HTML,
    Box,
    Button,
    GridspecLayout,
    HBox,
    Label,
    Layout,
    Output,
    Tab,
    Textarea,
    VBox,
)
from ruamel.yaml import YAML

from qcodes.dataset import experiments, initialise_or_create_database_at, plot_dataset

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from qcodes.dataset.data_set_protocol import DataSetProtocol
    from qcodes.dataset.descriptions.param_spec import ParamSpecBase

_META_DATA_KEY = "widget_notes"


def _get_in(nested_keys: Sequence[str], dct: dict[str, Any]) -> dict[str, Any]:
    """ Returns dct[i0][i1]...[iX] where [i0, i1, ..., iX]==nested_keys."""
    return reduce(operator.getitem, nested_keys, dct)


def button(
    description: str,
    button_style: str | None = None,
    on_click: Callable[[Any], None] | None = None,
    tooltip: str | None = None,
    layout_kwargs: dict[str, Any] | None = None,
    button_kwargs: dict[str, Any] | None = None,
) -> Button:
    """Returns a `ipywidgets.Button`."""
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


def button_to_text(title: str, body: str) -> Box:
    def _button_to_input(
        title: str, body: str, box: Box
    ) -> Callable[[Button], None]:
        def on_click(_: Button) -> None:
            text_input = Textarea(
                value=body,
                placeholder="Enter text",
                disabled=True,
                layout=Layout(height="300px", width="auto"),
            )
            back_button = button(
                "Back",
                "warning",
                on_click=_back_button(title, body, box),
                button_kwargs=dict(icon="undo"),
            )
            box.children = (text_input, back_button)

        return on_click

    def _back_button(
        title: str, body: str, box: Box
    ) -> Callable[[Button], None]:
        def on_click(_: Button) -> None:
            box.children = (_changeable_button(title, body, box),)

        return on_click

    def _changeable_button(title: str, body: str, box: Box) -> Button:
        return button(
            title, "success", on_click=_button_to_input(title, body, box),
        )

    box = VBox([], layout=Layout(height="auto", width="auto"))
    box.children = (_changeable_button(title, body, box),)
    return box


def label(description: str) -> Label:
    """Returns a `ipywidgets.Label` with text."""
    return Label(
        value=description, layout=Layout(height="max-content", width="auto")
    )


def _update_nested_dict_browser(
    nested_keys: Sequence[str], nested_dict: dict[Any, Any], box: Box
) -> Callable[[Button | None], None]:
    def update_box(_: Button | None) -> None:
        box.children = (_nested_dict_browser(nested_keys, nested_dict, box),)

    return update_box


def _nested_dict_browser(
    nested_keys: Sequence[str],
    nested_dict: dict[Any, Any],
    box: Box,
    max_nrows: int = 30,
) -> GridspecLayout:
    """Generates a `GridspecLayout` of the ``nested_keys`` in ``nested_dict``
    which is put inside of ``box``.

    Args:
        nested_keys: A sequence of keys of the nested dict. e.g., if
            ``nested_keys=['a', 'b']`` then ``nested_dict['a']['b']``.
        nested_dict: A dictionary that can contain more dictionaries as keys.
        box: An `ipywidgets.Box` instance.
        max_nrows: The maximum number of rows that can be displayed at once.
            Whenever the table has less than ``max_nrows`` rows, the table is
            displayed in 3 columns, otherwise it's 2 columns.
    """

    def _should_expand(x: Any) -> bool:
        return isinstance(x, dict) and x != {}

    col_widths = [8, 16, 30]
    selected_table = _get_in(nested_keys, nested_dict)
    nrows = (
        sum(
            len(v) if _should_expand(v) else 1 for v in selected_table.values()
        )
        + 1
    )
    ncols = 3

    if nrows > max_nrows:
        nrows = len(selected_table) + 1
        col_widths.pop(1)
        ncols = 2

    grid = GridspecLayout(nrows, col_widths[-1])
    update = partial(
        _update_nested_dict_browser, nested_dict=nested_dict, box=box
    )

    # Header
    title = " ► ".join(nested_keys)
    grid[0, :-1] = button(title, "success")
    up_click = update(nested_keys[:-1])
    grid[0, -1] = button("↰", "info", up_click)

    # Body

    row_index = 1
    for k, v in selected_table.items():
        row_length = len(v) if _should_expand(v) and ncols == 3 else 1
        but = button(k, "info", up_click)
        grid[row_index : row_index + row_length, : col_widths[0]] = but
        if _should_expand(v):
            if ncols == 3:
                for k_, v_ in v.items():
                    but = button(k_, "danger", update([*nested_keys, k]))
                    grid[row_index, col_widths[0] : col_widths[1]] = but
                    if _should_expand(v_):
                        sub_keys = ", ".join(v_.keys())
                        but = button(
                            sub_keys, "warning", update([*nested_keys, k, k_])
                        )
                    else:
                        but = label(str(v_))
                    grid[row_index, col_widths[1] :] = but
                    row_index += 1
            else:
                sub_keys = ", ".join(v.keys())
                grid[row_index, col_widths[0] :] = button(
                    sub_keys, "danger", update([*nested_keys, k])
                )
                row_index += 1
        else:
            grid[row_index, col_widths[0] :] = label(str(v))
            row_index += 1
    return grid


def nested_dict_browser(
    nested_dict: dict[Any, Any], nested_keys: Sequence[str] = ()
) -> Box:
    """Returns a widget to interactive browse a nested dictionary."""
    box = Box([])
    _update_nested_dict_browser(nested_keys, nested_dict, box)(None)
    return box


def _plot_ds(ds: DataSetProtocol) -> None:
    import matplotlib.pyplot as plt

    plot_dataset(ds)  # might fail
    plt.show()


def _do_in_tab(
    tab: Tab, ds: DataSetProtocol, which: Literal["plot", "snapshot"]
) -> Callable[[Button], None]:
    """Performs an operation inside of a subtab of a `ipywidgets.Tab`.

    Args:
        tab: Instance of `ipywidgets.Tab`.
        ds: A qcodes.DataSet instance.
        which: Either "plot" or "snapshot".
    """
    from IPython.display import clear_output, display
    assert which in ("plot", "snapshot")

    def delete_tab(output: Output, tab: Tab) -> Callable[[Button], None]:
        def on_click(_: Button) -> None:
            tab.children = tuple(c for c in tab.children if c != output)

        return on_click

    def _on_click(_: Button) -> None:
        assert which in ("plot", "snapshot")
        title = f"RID #{ds.captured_run_id} {which}"
        i = next(
            (i for i in range(len(tab.children)) if tab.get_title(i) == title),
            None,
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

            close_button = button(
                f"Close {which}",
                "danger",
                on_click=delete_tab(out, tab),
                button_kwargs=dict(icon="eraser"),
            )
            display(close_button)

            try:
                if which == "plot":
                    _plot_ds(ds)
                elif which == "snapshot":
                    snapshot = ds.snapshot
                    if snapshot is not None:
                        display(nested_dict_browser(snapshot))
                    else:
                        print("This dataset has no snapshot")
            except Exception:
                traceback.print_exc()
        tab.selected_index = i

    return _on_click


def create_tab(do_display: bool = True) -> Tab:
    """Creates a `ipywidgets.Tab` which can display outputs in its tabs."""
    from IPython.display import display
    output = Output()
    tab = Tab(children=(output,))

    tab.set_title(0, "Info")
    if do_display:
        display(tab)

    with output:
        # Prints it in the Output inside the tab.
        print("Plots and snapshots will show up here!")
    return tab


def editable_metadata(ds: DataSetProtocol) -> Box:
    def _button_to_input(text: str, box: Box) -> Callable[[Button], None]:
        def on_click(_: Button) -> None:
            text_input = Textarea(
                value=text,
                placeholder="Enter text",
                disabled=False,
                layout=Layout(height="auto", width="auto"),
            )
            save_button = button(
                "",
                "success",
                on_click=_save_button(box, ds),
                button_kwargs=dict(icon="save"),
                layout_kwargs=dict(width="50%"),
            )
            cancel_button = button(
                "",
                "danger",
                on_click=_save_button(box, ds, do_save=False),
                button_kwargs=dict(icon="close"),
                layout_kwargs=dict(width="50%"),
            )
            subbox = HBox([save_button, cancel_button],)
            box.children = (text_input, subbox)

        return on_click

    def _save_button(
        box: Box, ds: DataSetProtocol, do_save: bool = True
    ) -> Callable[[Button], None]:
        def on_click(_: Button) -> None:
            text = box.children[0].value
            if do_save:
                ds.add_metadata(tag=_META_DATA_KEY, metadata=text)
            box.children = (_changeable_button(text, box),)

        return on_click

    def _changeable_button(text: str, box: Box) -> Button:
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


def _yaml_dump(dct: dict[str, Any]) -> str:
    with io.StringIO() as f:
        YAML().dump(dct, f)
        return f.getvalue()


def _get_parameters(ds: DataSetProtocol) -> dict[str, dict[str, Any]]:
    independent = {}
    dependent = {}

    def _get_attr(p: ParamSpecBase) -> dict[str, Any]:
        return {
            "unit": p.unit,
            "label": p.label,
            "type": p.type,
        }

    for dep, indeps in ds.description.interdeps.dependencies.items():
        attrs = _get_attr(dep)
        attrs["depends_on"] = [p.name for p in indeps]
        dependent[dep.name] = attrs
        for p in indeps:
            independent[p.name] = _get_attr(p)
    return {"independent": independent, "dependent": dependent}


def _get_experiment_button(ds: DataSetProtocol) -> Box:
    title = f"{ds.exp_name}, {ds.sample_name}"
    body = _yaml_dump(
        {
            ".exp_name": ds.exp_name,
            ".sample_name": ds.sample_name,
            ".name": ds.name,
            ".path_to_db": ds.path_to_db,
            ".exp_id": ds.exp_id,
        }
    )
    return button_to_text(title, body)


def _get_timestamp_button(ds: DataSetProtocol) -> Box:
    start_timestamp = ds.run_timestamp_raw
    end_timestamp = ds.completed_timestamp_raw
    if start_timestamp is not None and end_timestamp is not None:
        total_time = str(
            datetime.fromtimestamp(end_timestamp)
            - datetime.fromtimestamp(start_timestamp)
        )
    else:
        total_time = "?"
    start = ds.run_timestamp()
    body = _yaml_dump(
        {
            ".run_timestamp": start,
            ".completed_timestamp": ds.completed_timestamp(),
            "total_time": total_time,
        }
    )
    return button_to_text(start or "", body)


def _get_run_id_button(ds: DataSetProtocol) -> Box:
    title = str(ds.run_id)
    body = _yaml_dump(
        {
            ".guid": ds.guid,
            ".captured_run_id": ds.captured_run_id,
            ".run_id": ds.run_id,
        }
    )
    return button_to_text(title, body)


def _get_parameters_button(ds: DataSetProtocol) -> Box:
    parameters = _get_parameters(ds)
    title = ds._parameters or ""
    return button_to_text(title, _yaml_dump(parameters))


def _get_snapshot_button(ds: DataSetProtocol, tab: Tab) -> Button:
    return button(
        "",
        "warning",
        tooltip="Click to open this DataSet's snapshot in a tab above.",
        on_click=_do_in_tab(tab, ds, "snapshot"),
        button_kwargs=dict(icon="camera"),
    )


def _get_plot_button(ds: DataSetProtocol, tab: Tab) -> Button:
    return button(
        "",
        "warning",
        tooltip="Click to open this DataSet's plot in a tab above.",
        on_click=_do_in_tab(tab, ds, "plot"),
        button_kwargs=dict(icon="line-chart"),
    )


def _experiment_widget(
    data_sets: Iterable[DataSetProtocol], tab: Tab
) -> GridspecLayout:
    """Show a `ipywidgets.GridspecLayout` with information about the
    loaded experiment. The clickable buttons can perform an action in ``tab``.
    """
    header_names = [
        "Run ID",
        "Experiment",
        "Name",
        "Parameters",
        "MSMT Time",
        "Notes",
        "Snapshot",
        "Plot",
    ]

    header = {n: button(n, "info") for n in header_names}
    rows = [header]
    for ds in data_sets:
        row = {}
        row["Run ID"] = _get_run_id_button(ds)
        row["Experiment"] = _get_experiment_button(ds)
        row["Name"] = label(ds.name)
        row["Notes"] = editable_metadata(ds)
        row["Parameters"] = _get_parameters_button(ds)
        row["MSMT Time"] = _get_timestamp_button(ds)
        row["Snapshot"] = _get_snapshot_button(ds, tab)
        row["Plot"] = _get_plot_button(ds, tab)
        rows.append(row)

    grid = GridspecLayout(n_rows=len(rows), n_columns=len(header_names))

    empty_label = label("")
    for i, row in enumerate(rows):
        for j, name in enumerate(header_names):
            grid[i, j] = row.get(name, empty_label)

    grid.layout.grid_template_rows = "auto " * len(rows)
    grid.layout.grid_template_columns = "auto " * len(header_names)
    return grid


def experiments_widget(
    db: str | None = None,
    data_sets: Sequence[DataSetProtocol] | None = None,
    *,
    sort_by: Literal["timestamp", "run_id"] | None = "run_id",
) -> VBox:
    r"""Displays an interactive widget that shows the :func:`qcodes.dataset.experiments`.

    With the edit button in the column ``Notes`` one can make persistent
    changes to the :class:`qcodes.dataset.DataSetProtocol`\s attribute
    ``metadata`` in the key "widget_notes".
    Expanding the coordinates or variables buttons, reveals more options, such
    as plotting or the ability to easily browse
    the :class:`qcodes.dataset.DataSetProtocol`\s snapshot.

    Args:
        db: Optionally pass a database file, if no database has been loaded.
        data_sets: Sequence of :class:`qcodes.dataset.DataSetProtocol`\s.
            If datasets are explicitly provided via this argument, the ``db``
            argument has no effect.
        sort_by: Sort datasets in widget by either "timestamp" (newest first),
            "run_id" or None (no predefined sorting).
    """
    if data_sets is None:
        if db is not None:
            initialise_or_create_database_at(db)
        data_sets = [ds for exp in experiments() for ds in exp.data_sets()]
    if sort_by == "run_id":
        data_sets = sorted(data_sets, key=lambda ds: ds.run_id)
    elif sort_by == "timestamp":
        data_sets = sorted(
            data_sets,
            key=lambda ds: ds.run_timestamp_raw if ds.run_timestamp_raw is not None else 0,
            reverse=True
        )

    title = HTML("<h1>QCoDeS experiments widget</h1>")
    tab = create_tab(do_display=False)
    grid = _experiment_widget(data_sets, tab)
    return VBox([title, tab, grid])


__all__ = ["experiments_widget", "nested_dict_browser"]
