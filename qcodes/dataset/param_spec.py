from qcodes.instrument.parameter import _BaseParameter


# TODO: we should validate type somehow
# we can't accept everything (or we can but crash at runtime?)
# we only support the types in VALUES type
class ParamSpec():
    def __init__(self, name: str, type: str, **metadata) -> None:
        self.name = name
        self.type = type
        if metadata:
            self.metadata = metadata

    def sql_repr(self):
        return f"{self.name} {self.type}"

    def __repr__(self):
        return f"{self.name} ({self.type})"


def param_spec(parameter: _BaseParameter, type: str) -> ParamSpec:
    """ Generates a ParamSpec from a qcodes parameter

    Args:
        - parameter: the qcodes parameter to make a spec

    """
    return ParamSpec(parameter.name, type, **parameter.metadata)
