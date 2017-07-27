from qcodes import Instrument
from qcodes.utils.validators import Numbers, Enum, Ints

import socket


class VTI_FQT(Instrument):
	"""
	Driver for the VTI fridge in FQT
	"""
	def __init__(self, name, address, port, **kwargs):
		super().__init__(name, **kwargs)
		self._address=address
		self._port=port

		self.add_parameter('temperature',
			unit='K',
			get_cmd=lambda:self.ask("TMP"),
			get_parser=float)

		self.add_parameter('helium_level',
			unit='mm',
			get_cmd=lambda:self.ask("HEL"),
			get_parser=float)




	def ask(self, cmd):
		BUFFER_SIZE = 10
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.connect((self._address, self._port))
		s.sendall(cmd.encode('utf-8'))
		data = s.recv(BUFFER_SIZE)
		s.close()
		return float(data)