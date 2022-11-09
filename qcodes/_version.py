def _get_version() -> str:
    import sys

    if sys.version_info >= (3, 9):
        from importlib.resources import files
    else:
        from importlib_resources import files

    import versioningit

    root_module = __loader__.name.split(".")[0]

    qcodes_path = files(root_module)
    return versioningit.get_version(project_dir=qcodes_path.parent)


__version__ = _get_version()
