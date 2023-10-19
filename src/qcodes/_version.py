def _get_version() -> str:
    from importlib.resources import files
    from pathlib import Path

    import versioningit

    # https://github.com/python/mypy/issues/4182
    root_module = __loader__.name.split(".")[0]  # type: ignore[name-defined]

    module_path = files(root_module)
    if isinstance(module_path, Path):
        return versioningit.get_version(project_dir=Path(module_path).parent.parent)
    else:
        return "0.0"


__version__ = _get_version()
