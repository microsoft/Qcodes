import time
from unittest.mock import patch

# importing ipykernel has the side effect
# of registering that as a kernel backend
# making the tests runnable
import ipykernel.ipkernel  # noqa  F401
import matplotlib
import pytest
from ipywidgets import (  # type: ignore[import-untyped]
    HTML,
    Button,
    GridspecLayout,
    Tab,
    Textarea,
)

from qcodes import interactive_widget

# set matplotlib backend before importing pyplot
matplotlib.use("Agg")


@pytest.fixture(name="tab", scope="function")
def _create_tab():
    yield interactive_widget.create_tab()


def test_snapshot_browser() -> None:
    dct = {"a": {"b": "c", "d": {"e": "f"}}}
    interactive_widget.nested_dict_browser(dct)
    interactive_widget.nested_dict_browser(dct, ["a"])


@pytest.mark.usefixtures("empty_temp_db")
def test_full_widget_on_empty_db() -> None:
    interactive_widget.experiments_widget()


@pytest.mark.usefixtures("experiment")
def test_full_widget_on_empty_experiment() -> None:
    interactive_widget.experiments_widget()


@pytest.mark.usefixtures("dataset")
def test_full_widget_on_empty_dataset() -> None:
    interactive_widget.experiments_widget()


@pytest.mark.usefixtures("standalone_parameters_dataset")
def test_full_widget_on_one_dataset() -> None:
    interactive_widget.experiments_widget()


def test_button_to_text(
    standalone_parameters_dataset,
) -> None:
    box = interactive_widget.button_to_text("title", "body")
    (button,) = box.children
    button.click()
    time.sleep(0.5)  # after click
    text_area, back_button = box.children
    assert "body" in text_area.value
    back_button.click()
    time.sleep(0.5)  # after click
    assert len(box.children) == 1


def test_snapshot_button(tab, standalone_parameters_dataset) -> None:
    ds = standalone_parameters_dataset
    snapshot_button = interactive_widget._get_snapshot_button(ds, tab)
    snapshot_button.click()
    time.sleep(0.5)  # after click
    # maybe use https://github.com/jupyter-widgets/ipywidgets/issues/2417
    assert "snapshot" in tab.get_title(1)


@patch("matplotlib.pyplot.show")
def test_plot_button(tab, standalone_parameters_dataset) -> None:
    ds = standalone_parameters_dataset
    plot_button = interactive_widget._get_plot_button(ds, tab)
    plot_button.click()
    time.sleep(0.5)  # after click


@pytest.mark.parametrize(
    "get_button_function",
    [
        interactive_widget._get_experiment_button,
        interactive_widget._get_timestamp_button,
        interactive_widget._get_run_id_button,
        interactive_widget._get_parameters_button,
    ],
)
def test_get_experiment_button(
    get_button_function, standalone_parameters_dataset,
) -> None:
    ds = standalone_parameters_dataset
    box = get_button_function(ds)
    snapshot_button = box.children[0]
    snapshot_button.click()
    time.sleep(0.5)  # after click
    assert len(box.children) == 2


def test_get_parameters(standalone_parameters_dataset) -> None:
    parameters = interactive_widget._get_parameters(
        standalone_parameters_dataset
    )
    assert bool(parameters["dependent"])  # not empty
    assert bool(parameters["independent"])  # not empty


def test_editable_metadata(
    standalone_parameters_dataset,
) -> None:
    ds = standalone_parameters_dataset
    box = interactive_widget.editable_metadata(ds)
    button = box.children[0]
    button.click()
    assert len(box.children) == 2
    text_area, save_box = box.children
    save_button = save_box.children[0]
    assert isinstance(text_area, Textarea)
    assert isinstance(button, Button)
    test_test = "test value"
    text_area.value = test_test
    save_button.click()
    time.sleep(0.5)  # after click
    # Test if metadata is saved.
    assert ds.metadata[interactive_widget._META_DATA_KEY] == test_test

    assert box.children[0].description == test_test


def test_experiments_widget(standalone_parameters_dataset) -> None:
    dss = [standalone_parameters_dataset]
    widget = interactive_widget.experiments_widget(data_sets=dss)
    assert len(widget.children) == 3
    html, tab, grid = widget.children
    assert isinstance(html, HTML)
    assert isinstance(tab, Tab)
    assert isinstance(grid, GridspecLayout)
    assert grid.n_rows == 1 + 1


@pytest.mark.parametrize('sort_by', [None, "run_id", "timestamp"])
def test_experiments_widget_sorting(standalone_parameters_dataset, sort_by) -> None:
    dss = [standalone_parameters_dataset]
    widget = interactive_widget.experiments_widget(
        data_sets=dss, sort_by=sort_by
    )
    assert len(widget.children) == 3
    grid = widget.children[2]
    assert isinstance(grid, GridspecLayout)
    assert grid.n_rows == 1 + 1
