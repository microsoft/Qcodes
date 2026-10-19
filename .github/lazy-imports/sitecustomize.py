"""Limit globally enabled lazy imports to imports involving QCoDeS."""

# ruff: noqa: INP001

import sys

_EAGER_LEGACY_MODULES = (
    "qcodes.dataset.dond",
    "qcodes.instrument_drivers.basel.BaselSP983",
    "qcodes.instrument_drivers.basel.BaselSP983a",
    "qcodes.instrument_drivers.basel.BaselSP983c",
    "qcodes.instrument_drivers.stanford_research.SG384",
    "qcodes.instrument_drivers.stanford_research.SR560",
    "qcodes.instrument_drivers.stanford_research.SR830",
    "qcodes.instrument_drivers.stanford_research.SR860",
    "qcodes.instrument_drivers.stanford_research.SR865",
    "qcodes.instrument_drivers.stanford_research.SR865A",
    "qcodes.instrument_drivers.stanford_research.SR86x",
    "qcodes.utils.full_class",
)


def qcodes_import_filter(
    importing: str | None, imported: str, fromlist: tuple[str, ...] | None
) -> bool:
    if any(
        imported == module or imported.startswith(f"{module}.")
        for module in _EAGER_LEGACY_MODULES
    ):
        return False
    return imported.startswith("qcodes") or (
        importing is not None and importing.startswith("qcodes")
    )


sys.set_lazy_imports_filter(qcodes_import_filter)
