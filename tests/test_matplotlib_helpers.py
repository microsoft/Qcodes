from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from qcodes.plotting import (
    apply_color_scale_limits,
    auto_color_scale_from_config,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from matplotlib.colorbar import Colorbar


@pytest.fixture(name="image_colorbar")
def _make_image_colorbar() -> Generator[Colorbar, None, None]:
    """A colorbar whose mappable is an ``AxesImage`` rather than a ``QuadMesh``."""
    fig, ax = plt.subplots()
    image = ax.imshow(np.arange(4).reshape(2, 2))
    yield fig.colorbar(image, ax=ax)
    plt.close(fig)


def test_apply_color_scale_limits_requires_mesh_data(image_colorbar: Colorbar) -> None:
    """Only ``QuadMesh`` mappables (as produced by ``pcolormesh``) can be
    rescaled; anything else is rejected with a clear error."""
    with pytest.raises(RuntimeError, match="Can only scale mesh data"):
        apply_color_scale_limits(image_colorbar, (0.0, 1.0))


def test_auto_color_scale_from_config_without_colorbar(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Scaling without a colorbar is a no-op that warns the user."""
    no_colorbar: Any = None

    with caplog.at_level(logging.WARNING, logger="qcodes.plotting.matplotlib_helpers"):
        auto_color_scale_from_config(no_colorbar, auto_color_scale=True)

    assert "did not receive a colorbar" in caplog.text
