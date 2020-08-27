import time

import pytest
from ipywidgets import Textarea, Button

# we only need `experiment` here, but pytest does not discover the dependencies
# by itself so we also need to import all the fixtures this one is dependent
# on
# pylint: disable=unused-import
from qcodes.tests.dataset.conftest import (
    dataset,
    experiment,
    empty_temp_db,
    standalone_parameters_dataset,
)
from qcodes import interactive_widget


@pytest.fixture(scope="function")
def tab():
    yield interactive_widget.create_tab()


def test_snapshot_browser():
    dct = {"a": {"b": "c", "d": {"e": "f"}}}
    interactive_widget.nested_dict_browser(dct)
    interactive_widget.nested_dict_browser(dct, ["a"])


@pytest.mark.usefixtures("experiment")
def test_full_widget():
    interactive_widget.experiments_widget()


def test_button_to_text(
    tab, standalone_parameters_dataset
):  # pylint: disable=redefined-outer-name
    box = interactive_widget.button_to_text("title", "body")
    (button,) = box.children
    button.click()
    time.sleep(1)  # after click
    text_area, back_button = box.children
    assert "body" in text_area.value
    back_button.click()
    time.sleep(1)  # after click
    assert len(box.children) == 1


def test_snapshot_button(
    tab, standalone_parameters_dataset
):  # pylint: disable=redefined-outer-name
    ds = standalone_parameters_dataset
    snapshot_button = interactive_widget._get_snapshot_button(ds, tab)
    snapshot_button.click()
    time.sleep(1)  # after click
    # maybe use https://github.com/jupyter-widgets/ipywidgets/issues/2417
    assert "snapshot" in tab.get_title(1)


def test_editable_metadata(standalone_parameters_dataset):  # pylint: disable=redefined-outer-name
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
    time.sleep(1)  # after click
    # Test if metadata is saved.
    assert ds.metadata[interactive_widget._META_DATA_KEY] == test_test

    assert box.children[0].description == test_test
