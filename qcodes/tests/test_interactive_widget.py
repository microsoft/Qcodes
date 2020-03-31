import pytest
import qcodes

# we only need `experiment` here, but pytest does not discover the dependencies
# by itself so we also need to import all the fixtures this one is dependent
# on
# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import (
    dataset,
    experiment,
    empty_temp_db,
)
from qcodes.tests.dataset.dataset_fixtures import standalone_parameters_dataset
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


def test_expandable_dict(
    standalone_parameters_dataset, tab
):  # pylint: disable=redefined-outer-name
    dct = {"a": {"b": "c", "d": {"e": "f"}}}
    interactive_widget.expandable_dict(dct, dataset, tab)
