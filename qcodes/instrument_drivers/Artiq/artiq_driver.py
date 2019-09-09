"""
	Driver for the Zotino and Sampler, CPU by V.Schmitt (May 2019)
	Modified by R.Savytskyy and M.Johnson (June 2019)
"""

import socket
from qcodes import Instrument


class Zotino(Instrument):

	def __init__(self, name, channel_dict, address, port, **kwargs):
		super().__init__(name, **kwargs)
		self.channel_dict = channel_dict
		self.address = address
		self.port = port
		for channel_name, channel_properties in self.channel_dict.items():
			self.add_parameter(name=channel_name, unit='V',
							   set_cmd=lambda x, ch=channel_properties['channel']: self.write("0 " + str(ch) + ' ' + str(x)),
							   get_cmd=lambda ch=channel_properties['channel']: self.ask("1 " + str(ch))
							   )

	def write(self, cmd):
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.connect((self.address, self.port))
		s.sendall(cmd.encode('utf-8'))
		s.close()

	def ask(self, cmd):
		BUFFER_SIZE = 10
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.connect((self.address, self.port))
		s.sendall(cmd.encode('utf-8'))
		data = s.recv(BUFFER_SIZE)
		s.close()
		return float(data)


class Sampler(Instrument):

	def __init__(self, name, channel_dict, address, port, **kwargs):
		super().__init__(name, **kwargs)
		self.channel_dict = channel_dict
		self.address = address
		self.port = port
		for channel_name, channel_properties in self.channel_dict.items():
			self.add_parameter(name=channel_name, unit='V',
							   get_cmd=lambda ch=channel_properties['channel'], average=channel_properties['average']: self.ask("3 " + str(ch) + " " + str(average))
							   )

	def ask(self, cmd):
		BUFFER_SIZE = 10
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.connect((self.address, self.port))
		s.sendall(cmd.encode('utf-8'))
		data = s.recv(BUFFER_SIZE)
		s.close()
		return float(data)