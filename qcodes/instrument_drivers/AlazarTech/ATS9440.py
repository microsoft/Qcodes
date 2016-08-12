from .ATS import AlazarTech_ATS, AlazarParameter
from qcodes.utils import validators


class ATS9440(AlazarTech_ATS):
    def __init__(self, name, dll_path, server_name=None):
        super().__init__(name, dll_path=dll_path)