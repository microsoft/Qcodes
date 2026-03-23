from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from qcodes.dataset.json_exporter import (
    export_data_as_json_heatmap,
    export_data_as_json_linear,
    json_template_heatmap,
    json_template_linear,
)


def test_json_template_linear_structure() -> None:
    assert json_template_linear["type"] == "linear"
    assert "x" in json_template_linear
    assert "y" in json_template_linear
    assert isinstance(json_template_linear["x"], dict)
    assert isinstance(json_template_linear["y"], dict)
    assert "data" in json_template_linear["x"]
    assert "data" in json_template_linear["y"]
    assert json_template_linear["x"]["is_setpoint"] is True
    assert json_template_linear["y"]["is_setpoint"] is False


def test_json_template_heatmap_structure() -> None:
    assert json_template_heatmap["type"] == "heatmap"
    assert "x" in json_template_heatmap
    assert "y" in json_template_heatmap
    assert "z" in json_template_heatmap
    assert isinstance(json_template_heatmap["x"], dict)
    assert isinstance(json_template_heatmap["y"], dict)
    assert isinstance(json_template_heatmap["z"], dict)
    assert json_template_heatmap["x"]["is_setpoint"] is True
    assert json_template_heatmap["y"]["is_setpoint"] is True
    assert json_template_heatmap["z"]["is_setpoint"] is False


def test_export_linear_writes_correct_json(tmp_path: Path) -> None:
    location = str(tmp_path / "linear.json")
    state: dict = {"json": copy.deepcopy(json_template_linear)}
    data = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]

    export_data_as_json_linear(data, len(data), state, location)

    with open(location) as f:
        result = json.load(f)

    assert result["type"] == "linear"
    assert result["x"]["data"] == [1.0, 2.0, 3.0]
    assert result["y"]["data"] == [10.0, 20.0, 30.0]


def test_export_linear_accumulates_data(tmp_path: Path) -> None:
    location = str(tmp_path / "linear.json")
    state: dict = {"json": copy.deepcopy(json_template_linear)}

    export_data_as_json_linear([[1.0, 10.0]], 1, state, location)
    export_data_as_json_linear([[2.0, 20.0]], 2, state, location)

    with open(location) as f:
        result = json.load(f)

    assert result["x"]["data"] == [1.0, 2.0]
    assert result["y"]["data"] == [10.0, 20.0]


def test_export_linear_does_nothing_for_empty_data(tmp_path: Path) -> None:
    location = str(tmp_path / "linear.json")
    state: dict = {"json": copy.deepcopy(json_template_linear)}

    export_data_as_json_linear([], 0, state, location)

    assert not Path(location).exists()


def test_export_heatmap_writes_correct_json(tmp_path: Path) -> None:
    location = str(tmp_path / "heatmap.json")
    xlen = 2
    ylen = 3
    total = xlen * ylen

    state: dict = {
        "json": copy.deepcopy(json_template_heatmap),
        "data": {
            "x": np.zeros(total),
            "y": np.zeros(total),
            "z": np.zeros(total),
            "location": 0,
            "xlen": xlen,
            "ylen": ylen,
        },
    }

    # 2x3 grid: x varies slowly, y varies fast
    data = [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 2.0],
        [0.0, 2.0, 3.0],
        [1.0, 0.0, 4.0],
        [1.0, 1.0, 5.0],
        [1.0, 2.0, 6.0],
    ]

    export_data_as_json_heatmap(data, total, state, location)

    with open(location) as f:
        result = json.load(f)

    assert result["type"] == "heatmap"
    assert result["x"]["data"] == [0.0, 1.0]
    assert result["y"]["data"] == [0.0, 1.0, 2.0]
    assert result["z"]["data"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_export_heatmap_does_nothing_for_empty_data(tmp_path: Path) -> None:
    location = str(tmp_path / "heatmap.json")
    state: dict = {
        "json": copy.deepcopy(json_template_heatmap),
        "data": {
            "x": np.zeros(4),
            "y": np.zeros(4),
            "z": np.zeros(4),
            "location": 0,
            "xlen": 2,
            "ylen": 2,
        },
    }

    export_data_as_json_heatmap([], 0, state, location)

    assert not Path(location).exists()
