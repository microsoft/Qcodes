def _get_version() -> str:
    import sys

    if sys.version_info >= (3, 9):
        from importlib.resources import as_file, files
    else:
        from importlib_resources import as_file, files

    import versioningit

    qcodes_path = files("qcodes")
    return versioningit.get_version(project_dir=qcodes_path.parent)


__version__ = _get_version()
