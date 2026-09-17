from __future__ import annotations

import logging
from typing import Any

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from qcodes.plotting import (
    apply_color_scale_limits,
    auto_color_scale_from_config,
)


def test_apply_color_scale_limits_requires_mesh_data() -> None:
    """Only ``QuadMesh`` mappables (as produced by ``pcolormesh``) can be
    rescaled; anything else is rejected with a clear error."""
    fig, ax = plt.subplots()
    try:
        image = ax.imshow(np.arange(4).reshape(2, 2))
        colorbar = fig.colorbar(image, ax=ax)

        with pytest.raises(RuntimeError, match="Can only scale mesh data"):
            apply_color_scale_limits(colorbar, (0.0, 1.0))
    finally:
        plt.close(fig)


def test_auto_color_scale_from_config_without_colorbar(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Scaling without a colorbar is a no-op that warns the user."""
    no_colorbar: Any = None

    with caplog.at_level(logging.WARNING, logger="qcodes.plotting.matplotlib_helpers"):
        auto_color_scale_from_config(no_colorbar, auto_color_scale=True)

    assert "did not receive a colorbar" in caplog.text
