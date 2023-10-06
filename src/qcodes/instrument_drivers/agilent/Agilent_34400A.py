from qcodes.utils import deprecate

from ._Agilent_344xxA import _Agilent344xxA


@deprecate(
    alternative="Keysight 344xxA drivers or Agilent34401A, Agilent34410A, Agilent34411A"
)
class Agilent_34400A(_Agilent344xxA):
    pass
