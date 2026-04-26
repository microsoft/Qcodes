import os
from pathlib import Path

# Drivers that require optional third-party libraries not generally available.
# These must be excluded from doctest collection to avoid ImportErrors.
collect_ignore_glob = [
    os.path.join("Galil", "*"),
    os.path.join("Minicircuits", "*"),
    os.path.join("QuantumDesign", "DynaCoolPPMS", "private", "*"),
]


def _find_deprecated_driver_modules() -> list[str]:
    """
    Scan for deprecated driver alias modules that emit deprecation warnings
    on import. Importing these during doctest collection triggers noisy
    warnings and serves no purpose since they contain no doctests.
    """
    drivers_dir = Path(__file__).parent
    deprecated: list[str] = []
    for path in sorted(drivers_dir.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if "module is deprecated" in text:
            deprecated.append(str(path.relative_to(drivers_dir)))
    return deprecated


collect_ignore = _find_deprecated_driver_modules()
