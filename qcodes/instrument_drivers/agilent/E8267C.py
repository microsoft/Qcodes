from qcodes.utils import deprecate

from .Agilent_E8267C import AgilentE8267C


@deprecate(alternative="AgilentE8267C")
class E8267(AgilentE8267C):
    pass
