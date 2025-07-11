def _get_version() -> str:
    # we use lazy imports to avoid importing modules that are not
    # used when the use of this function is patched out at build time
    from importlib.resources import files  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import versioningit  # noqa: PLC0415

    module_path = files("qcodes")
    if isinstance(module_path, Path):
        return versioningit.get_version(project_dir=Path(module_path).parent.parent)
    else:
        return "0.0"


__version__ = _get_version()
