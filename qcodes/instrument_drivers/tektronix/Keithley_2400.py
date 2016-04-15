# Keithley_2400.py driver for Keithley 2400 DMM
#
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

from qcodes.instrument.visa import VisaInstrument
import time
import logging

#%% Helper functions



#%% Driver for Keithley_2400

class Keithley_2400(VisaInstrument):
    '''
    TODO
    '''
    def __init__(self, name, address, reset=False, **kwargs):
        t0 = time.time()
        super().__init__(name, address, terminator='\n', **kwargs)

        self._modes = ['']
        self.add_parameter('IDN', get_cmd='*IDN?')

        # Add parameters to wrapper
        self.add_parameter('volt', get_cmd=':READ?',
                           get_parser=self.get_volt, set_cmd='sour:volt:lev {:.5f};',
                           units='V' )
        self.add_parameter('curr', get_cmd=':READ?',
                           get_parser=self.get_curr, set_cmd='sour:curr:lev {:.5f};',
                           units='A' )
        # One might want to initialize like this:
        # ':SOUR:VOLT:MODE FIX'
        # ':SENS:FUNC:ON "CURR"'

    def get_curr(self, msg):
        return float(msg.split(',')[1])

    def get_volt(self, msg):
        return float(msg.split(',')[0])

