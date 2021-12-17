"""
Tests for `qcodes.utils.plotting`.
"""
import matplotlib

# set matplotlib backend before importing pyplot
matplotlib.use("Agg")

from matplotlib import pyplot as plt
from pytest import fixture

import qcodes
from qcodes.dataset.plotting import plot_by_id
from qcodes.tests.common import default_config

from .dataset_generators import dataset_with_outliers_generator


@fixture(scope='function')
def dataset_with_data_outside_iqr_high_outlier(dataset):
    return dataset_with_outliers_generator(dataset, data_offset=2, low_outlier=-2,
                                           high_outlier=3)


@fixture(scope='function')
def dataset_with_data_outside_iqr_low_outlier(dataset):
    return dataset_with_outliers_generator(dataset, data_offset=2, low_outlier=-2,
                                           high_outlier=3)


@fixture(scope='function')
def dataset_with_outliers(dataset):
    return dataset_with_outliers_generator(dataset, low_outlier=-3,
                                           high_outlier=3,
                                           background_noise=False)


def test_extend(dataset_with_outliers):
    # this one should clipp the upper values
    run_id = dataset_with_outliers.run_id
    _, cb = plot_by_id(run_id, auto_color_scale=False)
    assert cb[0].extend == 'neither'
    _, cb = plot_by_id(run_id, auto_color_scale=True, cutoff_percentile=(0, 0.5))
    assert cb[0].extend == 'min'
    _, cb = plot_by_id(run_id, auto_color_scale=True, cutoff_percentile=(0.5, 0))
    assert cb[0].extend == 'max'
    plt.close()


def test_defaults(dataset_with_outliers):
    run_id = dataset_with_outliers.run_id

    # plot_by_id loads from the database location provided in the qcodes
    # config. But the tests are supposed to run with the standard config.
    # Therefore we need to backup the db location and add it to the default
    # config context.
    db_location = qcodes.config.core.db_location
    with default_config():
        qcodes.config.core.db_location = db_location
        _, cb = plot_by_id(run_id)
        assert cb[0].extend == 'neither'

        qcodes.config.plotting.auto_color_scale.enabled = True

        _, cb = plot_by_id(run_id)
        assert cb[0].extend == 'both'
    plt.close()
