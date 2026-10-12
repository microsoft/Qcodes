"""Limit globally enabled lazy imports to imports involving QCoDeS."""

import sys


def qcodes_import_filter(
    importing: str | None, imported: str, fromlist: tuple[str, ...] | None
) -> bool:
    return imported.startswith("qcodes") or (
        importing is not None and importing.startswith("qcodes")
    )


sys.set_lazy_imports_filter(qcodes_import_filter)
