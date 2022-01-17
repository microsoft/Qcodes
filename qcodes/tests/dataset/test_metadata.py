import pytest

from qcodes.tests.common import error_caused_by


def test_get_metadata_from_dataset(dataset):
    dataset.add_metadata('something', 123)
    something = dataset.get_metadata('something')
    assert 123 == something


def test_get_nonexisting_metadata(dataset):

    data = dataset.get_metadata("something")
    assert data is None
