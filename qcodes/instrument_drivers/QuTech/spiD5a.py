#%% Load packages
import time
import logging
import numpy as np
import traceback
import threading
from functools import partial

from qcodes import Instrument, validators as vals
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils.validators import Bool, Numbers

import sys

sys.path.append(r'D:\drivers\SPI-rack\spirack')

from spi_rack import SPI_rack
from D5a_module import D5a_module

#%%

class D5a(Instrument):
    """ Qcodes driver for the SPI D5a module
    
    """

    def __init__(self, name, address='COM5', spi_module=1, spi_baud=1000000, spi_timeout=1, reset=False, dac_step=10,
                 dac_delay=.1, dac_max_delay=0.2, divider=1000, safe_version=True,
                 use_locks=True, **kwargs):
        """
        Initialzes the D5a, and communicates with the wrapper

        Args:
            name (string)        : name of the instrument
            address (string)     : ASRL address
            reset (bool)         : resets to default values, default=false
            dac_step (float)         : max step size for dac parameter
            dac_delay (float)        : delay (in seconds) for dac
            dac_max_delay (float)    : maximum delay before emitting a warning
        """
        self.verbose=1
        self.spi_rack = SPI_rack(address, spi_baud, spi_timeout)
        self.spi_rack.unlock()

        self.D5a = D5a_module(self.spi_rack, spi_module)
        
        t0 = time.time()
        super().__init__(name, **kwargs)
        if use_locks:
            self.lock = threading.Lock()
        else:
            self.lock = None

        self._numdacs=16
        self.divider=divider
        self._dacoffset=1
        
        if divider==1000:
            unit='mV'
        elif divider==1:
            unit='V'
        else:
            unit='a.u.'

        for i in range(self._dacoffset, self._numdacs + self._dacoffset):
            self.add_parameter(
                'dac{}'.format(i),
                label='dac{} '.format(i),
                unit=unit,
                get_cmd=partial(self._get_voltage, i-self._dacoffset),
                set_cmd=partial(self._set_voltage, i-self._dacoffset),
                vals=vals.Numbers(-2000, 2000),
                step=dac_step, 
                delay=dac_delay, 
                max_delay=dac_max_delay, 
                max_val_age=10 
                )

        t1 = time.time()

        if self.verbose:
            print('Initialized %s in %.2fs' % (self.name, t1 - t0))


    def get_idn(self):
        """
        Overwrites the get_idn function using constants as the hardware
        does not have a proper \*IDN function.
        """

        idparts = ['QuTech', 'D5a', 'None', '?']

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))


    def get_all(self):
        return self.snapshot(update=True)

    def _get_voltages(self):        
        return self.D5a.voltages
    
    def _set_voltage(self, dacidx, value):
        #print('_get_voltage: idx %d, value %f' % (dacidx, v))
        if self.lock:
            with self.lock:
                self.D5a.set_voltage(dacidx, value/self.divider)
        else:
            self.D5a.set_voltage(dacidx, value/self.divider)
    def _get_voltage(self, dacidx):
        if self.lock:
            with self.lock:
                [voltage, span] = self.D5a.get_settings(dacidx)
        else:
            [voltage, span] = self.D5a.get_settings(dacidx)
        if self.verbose>=2:
            print('_get_voltage: idx %d, value %f %f' % (dacidx, voltage, span))
        return voltage*self.divider
        
#        v=self.D5a.voltages[dacidx]*self.divider
#        return v
        

    def set_dacs_zero(self):
        for i in range(self._numdacs):
            self.set(i + 1, 0)

    def close(self):
        if self.spi_rack is not None:
            self.spi_rack.close()
            self.spi_rack=None


    def _gen_ch_get_func(self, fun, ch):
        def get_func():
            return fun(ch)
        return get_func

#%% Testing

if __name__=='__main__':
    from qcodes.instrument_drivers.Spectrum.M4i import M4i
    import qcodes.instrument_drivers.tektronix.Keithley_2000 as keith2000

    keithley1 = keith2000.Keithley_2000('keithley1', 'GPIB0::15::INSTR')
    ivvi=D5a(name='spiivvi2', address='COM5')    
    digitizer = M4i(name='digitizer')
    digitizer.initialize_channels()
    digitizer.get_error_info32bit()
#%%
if __name__=='__main__':

    for ii in range(0,800,80):
        ivvi.dac1.set(ii/1000.)
        keithley1
        print('ivvi %f, keithley1.amplitude %f'  % (ivvi.dac1.get(), keithley1.amplitude.get() ) )
        #print('ivvi %f, digitizer.channel_3 %f'  % (ivvi.dac1.get(), digitizer.channel_3.get() ) )
        time.sleep(.05)
        

#%% Testing      
if 0:
    D5a=ivvi.D5a
    sweepvolts = np.linspace(-.1, .1, 50)
    
    i = 0
    try:
        while True:
            D5a.set_voltage(0, sweepvolts[i])
            print('i %d' % i)
            time.sleep(1e-2)
            i = (i + 1) % len(sweepvolts)
    except KeyboardInterrupt: # interrupt with ctrl-c
        pass
    

   