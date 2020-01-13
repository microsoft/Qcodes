import pytest

# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import dataset, experiment, \
    empty_temp_db
from qcodes.tests.common import error_caused_by


def test_get_metadata_from_dataset(dataset):
    dataset.add_metadata('something', 123)
    something = dataset.get_metadata('something')
    assert 123 == something


def test_get_nonexisting_metadata(dataset):
    with pytest.raises(RuntimeError) as excinfo:
        _ = dataset.get_metadata('something')
    assert error_caused_by(excinfo, "no such column: something")
