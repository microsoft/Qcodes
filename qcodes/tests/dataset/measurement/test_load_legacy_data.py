import json
from pathlib import Path

import pytest

from qcodes.dataset import import_dat_file, load_by_id
from qcodes.dataset.data_set import DataSet


@pytest.mark.usefixtures("experiment")
def test_load_legacy_files_2d() -> None:
    full_location = (
        Path(__file__).parent.parent
        / "fixtures"
        / "data_2018_01_17"
        / "data_002_2D_test_15_43_14"
    )
    run_ids = import_dat_file(str(full_location))
    run_id = run_ids[0]
    data = load_by_id(run_id)
    assert isinstance(data, DataSet)
    assert data.parameters == "dac_ch1_set,dac_ch2_set,dmm_voltage"
    assert data.number_of_results == 36
    expected_names = ["dac_ch1_set", "dac_ch2_set", "dmm_voltage"]
    expected_labels = ["Gate ch1", "Gate ch2", "Gate voltage"]
    expected_units = ["V", "V", "V"]
    expected_depends_on = ["", "", "dac_ch1_set, dac_ch2_set"]
    for i, parameter in enumerate(data.get_parameters()):
        assert parameter.name == expected_names[i]
        assert parameter.label == expected_labels[i]
        assert parameter.unit == expected_units[i]
        assert parameter.depends_on == expected_depends_on[i]
        assert parameter.type == "numeric"
    snapshot_str = data.get_metadata("snapshot")
    assert isinstance(snapshot_str, str)
    snapshot = json.loads(snapshot_str)
    assert sorted(list(snapshot.keys())) == [
        "__class__",
        "arrays",
        "formatter",
        "io",
        "location",
        "loop",
        "station",
    ]


@pytest.mark.usefixtures("experiment")
def test_load_legacy_files_1d() -> None:
    full_location = (
        Path(__file__).parent.parent
        / "fixtures"
        / "data_2018_01_17"
        / "data_001_testsweep_15_42_57"
    )
    run_ids = import_dat_file(str(full_location))
    run_id = run_ids[0]
    data = load_by_id(run_id)
    assert isinstance(data, DataSet)
    assert data.parameters == "dac_ch1_set,dmm_voltage"
    assert data.number_of_results == 201
    expected_names = ["dac_ch1_set", "dmm_voltage"]
    expected_labels = ["Gate ch1", "Gate voltage"]
    expected_units = ["V", "V"]
    expected_depends_on = ["", "dac_ch1_set"]
    for i, parameter in enumerate(data.get_parameters()):
        assert parameter.name == expected_names[i]
        assert parameter.label == expected_labels[i]
        assert parameter.unit == expected_units[i]
        assert parameter.depends_on == expected_depends_on[i]
        assert parameter.type == "numeric"
    snapshot_str = data.get_metadata("snapshot")
    assert isinstance(snapshot_str, str)
    snapshot = json.loads(snapshot_str)
    assert sorted(list(snapshot.keys())) == [
        "__class__",
        "arrays",
        "formatter",
        "io",
        "location",
        "loop",
        "station",
    ]


@pytest.mark.usefixtures("experiment")
def test_load_legacy_files_1d_pathlib_path() -> None:
    full_location = (
        Path(__file__).parent.parent
        / "fixtures"
        / "data_2018_01_17"
        / "data_001_testsweep_15_42_57"
    )
    run_ids = import_dat_file(full_location)
    run_id = run_ids[0]
    data = load_by_id(run_id)
    assert isinstance(data, DataSet)
    assert data.parameters == "dac_ch1_set,dmm_voltage"
    assert data.number_of_results == 201
    expected_names = ["dac_ch1_set", "dmm_voltage"]
    expected_labels = ["Gate ch1", "Gate voltage"]
    expected_units = ["V", "V"]
    expected_depends_on = ["", "dac_ch1_set"]
    for i, parameter in enumerate(data.get_parameters()):
        assert parameter.name == expected_names[i]
        assert parameter.label == expected_labels[i]
        assert parameter.unit == expected_units[i]
        assert parameter.depends_on == expected_depends_on[i]
        assert parameter.type == "numeric"
    snapshot_str = data.get_metadata("snapshot")
    assert isinstance(snapshot_str, str)
    snapshot = json.loads(snapshot_str)
    assert sorted(list(snapshot.keys())) == [
        "__class__",
        "arrays",
        "formatter",
        "io",
        "location",
        "loop",
        "station",
    ]
