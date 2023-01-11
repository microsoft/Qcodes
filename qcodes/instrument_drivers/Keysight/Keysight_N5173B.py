from qcodes.instrument_drivers.Keysight.Keysight_N5183B import KeysightN5183B


class KeysightN5173B(KeysightN5183B):
    pass  # N5173B has the same interface as N5183B


class N5173B(KeysightN5173B):
    """
    Alias for backwards compatibility
    """
