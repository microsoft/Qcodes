def _get_version() -> str:
    from pathlib import Path

    import versioningit

    import qcodes

    qcodes_path = Path(qcodes.__file__).parent
    return versioningit.get_version(project_dir=qcodes_path.parent)


__version__ = _get_version()
