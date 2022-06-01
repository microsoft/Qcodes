def _get_version() -> str:
    from pathlib import Path

    import versioningit

    return versioningit.get_version(project_dir=Path(__file__).parent.parent)


__version__ = _get_version()
