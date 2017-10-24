from time import sleep, time
import numpy as np
import ctypes as ct
import logging
import os

from qcodes import Instrument, validators as vals
from qcodes.instrument.parameter import ManualParameter


class LMS(Instrument):

	"""
	Driver for the Vaunix LabBrick generator - LMS series, UBS connected. 
	Device identification is made by Serial number

    Status: Beta version.
        This driver is functional but not all features have been implemented. Only continuous signals have been implemented, i.e. no pulsing

    TODO:
        Add pulsing functions

    Working on devices: 
     + LMS-103
     + 

	""" 

	dll_path = os.path.dirname(__file__)  +'\\vnx_fmsynth.dll'

	def __init__(self, name, serial_number, dll_path=None, **kwargs):
		super().__init__(name, **kwargs)

		self.serial_number=serial_number

		self.dll = ct.CDLL(dll_path or self.dll_path)

		self.initDevice()

		self.add_parameter('frequency',
							label='Frequency',
							unit='Hz',
							get_cmd=self.get_frequency,
							set_cmd=self.set_frequency,
							vals=vals.Numbers())

		self.add_parameter('power',
							label='Power',
							unit='dBm',
							get_cmd=self.get_power,
							set_cmd=self.set_power,
							vals=vals.Numbers())

		self.add_parameter('output',
							label='output',
							get_cmd=self.get_output_state,
							set_cmd=self.set_output_state,
							vals=vals.Bool())


	def initDevice(self):

		self.dll.fnLMS_SetTestMode(False) # non testing mode

		number_of_devices = int(self.dll.fnLMS_GetNumDevices())
		if number_of_devices == 0:
			raise IOError('No device was found!')

		active_devices=(ct.c_uint * number_of_devices)()

		self.dll.fnLMS_GetDevInfo(active_devices)

		self.deviceID = max((dev if self.serial_number == int(self.dll.fnLMS_GetSerialNumber(dev)) else 0) for dev in active_devices)
		if self.deviceID==0: 
			raise IOError('Device with S/N %i was not found!'%self.serial_number)

	def open(self):
		self.dll.fnLMS_InitDevice(self.deviceID)

	def close(self):
		self.dll.fnLMS_CloseDevice(self.deviceID)

	def get_frequency(self):
		self.open()
		freq=self.dll.fnLMS_GetFrequency(self.deviceID) 
		self.close()
		return float(freq) * 10

	def set_frequency(self,freq):
		self.open()
		self.dll.fnLMS_SetFrequency(self.deviceID, ct.c_int(int(freq/10))) 
		self.close()


	def get_power(self):
		self.open()
		power = float(self.dll.fnLMS_GetAbsPowerLevel(self.deviceID)/4.)
		self.close()
		return power

	def set_power(self,power):
		self.open()
		self.dll.fnLMS_SetPowerLevel(self.deviceID,int(4*power))
		self.close()

	def get_output_state(self):
		self.open()
		ret=self.dll.fnLMS_GetRF_On(self.deviceID)
		self.close()
		return bool(ret)
	
	def set_output_state(self, state):
		self.open()
		self.dll.fnLMS_SetRFOn(self.deviceID, int(state))
		self.close()
