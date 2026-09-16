from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from qcodes.plotting import auto_color_scale_from_config

if TYPE_CHECKING:
    import pytest


def test_auto_color_scale_from_config_without_colorbar(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Scaling without a colorbar is a no-op that warns the user."""
    no_colorbar: Any = None

    with caplog.at_level(logging.WARNING, logger="qcodes.plotting.matplotlib_helpers"):
        auto_color_scale_from_config(no_colorbar, auto_color_scale=True)

    assert "did not receive a colorbar" in caplog.text
