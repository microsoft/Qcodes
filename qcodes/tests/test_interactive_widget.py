import pytest
import qcodes
from qcodes.dataset.data_set import load_by_counter, new_data_set
from qcodes.tests.dataset.temporary_databases import experiment, empty_temp_db
from qcodes.utils import interactive_widget


@pytest.fixture(scope="function")
def tab():
    yield interactive_widget.create_tab()


@pytest.fixture(scope="function")
def dataset(experiment):
    exps = qcodes.experiments()
    exp = exps[0]
    yield load_by_counter(1, 1)


def test_snapshot_browser():
    dct = {"a": {"b": "c", "d": {"e": "f"}}}
    interactive_widget.nested_dict_browser(dct)
    interactive_widget.nested_dict_browser(dct, ["a"])


@pytest.mark.usefixtures("experiment")
def test_full_widget():
    interactive_widget.experiments_widget()


def test_expandable_dict(dataset, tab):
    dct = {"a": {"b": "c", "d": {"e": "f"}}}
    interactive_widget.expandable_dict(dct, dataset, tab)
