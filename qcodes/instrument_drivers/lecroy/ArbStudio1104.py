import os
import clr  # Import pythonnet to talk to dll
from System import Array
from time import sleep



class ArbStudio1104(Instrument):
    def __init__(self, name, dll_path, **kwargs):
        super().__init__(name, **kwargs)

        # Add .NET assembly to pythonnet
        # This allows importing its functions
        clr.System.Reflection.Assembly.LoadFile(dll_path)
        from clr import ActiveTechnologies
        self.api = ActiveTechnologies.Instruments.AWG4000.Control

        # Get device object
        self.device = api.DeviceSet().DeviceList[0]

        self.add_parameter('core_clock',
                           label='Core clock',
                           units='MHz',
                           set_cmd=api.pb_core_clock,
                           vals=Numbers(0, 500))

    def initialize(self):
        # Create empty array of four channels.
        # These are only necessary for initialization
        channels = Array.CreateInstance(self.api.Functionality, 4)
        # Initialize each of the channels
        channels[0] = self.api.Functionality.ARB
        channels[1] = self.api.Functionality.ARB
        channels[2] = self.api.Functionality.ARB
        channels[3] = self.api.Functionality.ARB

        # Initialise ArbStudio
        return_msg = self.device.Initialize(channels)
        assert return_msg.ErrorSource == self.api.ErrorCodes.RES_SUCCESS, \
            "Error initializing ARB: {}".format(return_msg.ErrorDescription)
        return return_msg