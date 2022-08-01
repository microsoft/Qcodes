def full_class(obj: object) -> str:
    """The full importable path to an object's class."""
    return type(obj).__module__ + "." + type(obj).__name__
