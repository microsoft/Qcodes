# mercuryiPS.py driver for Oxford MercuryiPS magnet power supply
#
# The MIT License (MIT)
# Copyright (c) 2016 Merlin von Soosten <merlin.von.soosten@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in theSoftware without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from functools import partial
import re

from qcodes import VisaInstrument
from qcodes.utils.validators import Strings, Enum


class MercuryiPS(VisaInstrument):
    '''
    MercuryiPS Driver

    This is the qcodes driver for the Oxford MercuryiPS magnet power supply.

    Status: beta-version.
        TODO:
        - SAFETY!! we need to make sure the magnet is only ramped at certain
          conditions!
        - Add parameters that get data for all channels:
          magnet.fld.get() should return [fx, fy, fz] or whatever axes are
          available
        - Fix this call = ''
                   eval(call)
          stuff, I guess there is a smarter way of doing that?
        - this findall stuff in _get_cmd, is that smart?
    '''
    def __init__(self, name, axes=None, **kwargs):
        super().__init__(name, terminator='\n', **kwargs)
        self.axes = axes
        self._ATOB = {}
        self._latest_response = ''
        # for some reason the first call is always invalid?! need some kind of init?
        self.ask('*IDN?')
        IDN = self.ask('*IDN?')[4:]
        vendor, model, serial, firmware = map(str.strip, IDN.split(':'))

        self.IDN = {'vendor': vendor, 'model': model,
                    'serial': serial, 'firmware': firmware}

        if axes is None:
            self._determine_magnet_axes()
        self._determine_current_to_field()

        for ax in self.axes:
            self.add_parameter(ax.lower()+'_fld',
                               get_cmd=partial(self._get_fld, ax, 'FLD'),
                               set_cmd=partial(self._ramp_to_setpoint, ax, 'FSET'),
                               label='B'+ax.lower(),
                               units='T')
            self.add_parameter(ax.lower()+'_fldC',
                               get_cmd=partial(self._get_fld_converted,
                                               ax, 'CURR'),
                               set_cmd=partial(self._ramp_to_setpoint, ax, 'CSET'),
                               label='B'+ax.lower(),
                               units='T')
            self.add_parameter(ax.lower()+'_ACTN',
                               get_cmd=partial(self._get_cmd,
                                               'READ:DEV:GRP{}:PSU:ACTN?'.format(ax)),
                               set_cmd='SET:DEV:GRP{}:PSU:ACTN:'.format(ax)+'{}',
                               vals=Enum('HOLD', 'RTOS', 'RTOZ', 'CLMP'))
            self.add_parameter(ax.lower()+'_setpoint',
                               get_cmd=partial(self._get_fld, ax, 'FSET'),
                               set_cmd=partial(self._set_fld, ax, 'FSET'),
                               units='T')
            self.add_parameter(ax.lower()+'_setpointC',
                               get_cmd=partial(self._get_fld_converted, ax, 'CSET'),
                               set_cmd=partial(self._set_fld_converted, ax, 'CSET'),
                               units='T')
            self.add_parameter(ax.lower()+'_rate',
                               get_cmd=partial(self._get_fld, ax, 'RFST'),
                               set_cmd=partial(self._set_fld, ax, 'RFST'),
                               units='T/m')
            self.add_parameter(ax.lower()+'_rateC',
                               get_cmd=partial(self._get_fld_converted, ax, 'RCST'),
                               set_cmd=partial(self._set_fld_converted, ax, 'RCST'),
                               units='T/m')
    def hold(self):
        for ax in self.axes:
            # How do I properly call those parameters from here?
            # self.{ax}_ACTN.set('HOLD')
            call = 'self.{}_ACTN.set("HOLD")'.format(ax.lower())
            eval(call)

    def to_zero(self):
        for ax in self.axes:
            # How do I properly call those parameters from here?
            # self.{ax}_ACTN.set('HOLD')
            call = 'self.{}_ACTN.set("RTOZ")'.format(ax.lower())
            eval(call)

    def _ramp_to_setpoint(self, ax, cmd, setpoint):
        # There should be a blocking and non-blocking version of this
        if cmd is 'CSET':
            self._set_fld_converted(ax, cmd, setpoint)
        elif cmd is 'FSET':
            self._set_fld(ax, cmd, setpoint)
        self.write('SET:DEV:GRP{}:PSU:ACTN:RTOS'.format(ax))

    def _set_fld(self, ax, cmd, setpoint):
        # Could be FSET for setpoint
        #          RFST for rate
        cmd = 'SET:DEV:GRP{}:PSU:SIG:{}:{:6f}'.format(ax, cmd, setpoint)
        self.write(cmd)

    def _get_fld(self, ax, cmd):
        # Could be FSET for setpoint
        #          FLD for field
        #          RFLD for rate
        #          PFLD persistent field reading
        fld = self._get_cmd('READ:DEV:GRP{}:PSU:SIG:{}?'.format(ax, cmd), float)
        return fld

    def _set_fld_converted(self, ax, cmd, setpoint):
        # Could be CSET for setpoint
        #          RCST for rate
        #
        # We set current, not field, this gives higher resolution due to
        # limited number of digits
        cur = setpoint * self._ATOB[ax]
        cmd = 'SET:DEV:GRP{}:PSU:SIG:{}:{:6f}'.format(ax, cmd, cur)
        self.write(cmd)

    def _get_fld_converted(self, ax, cmd):
        # Could be CSET for setpoint
        #          CURR for field
        #          RCUR for rate
        #          PCUR persistent current reading
        # We ask for current, not field, this gives higher resolution due to
        # limited number of digits
        # The conversion gives us a float with lots of digits, how to limit that?
        curr = self._get_cmd('READ:DEV:GRP{}:PSU:SIG:{}?'.format(ax, cmd), float)
        return curr / self._ATOB[ax]

    def _get_cmd(self, question, parser=None):
        msg = self.ask(question)[len(question):]
        # How would one macth this without specifying the units?
        # m = re.match('STAT:DEV:GRPX:PSU:SIG:RFST:(.+?)T/m',
        #              'STAT:DEV:GRPX:PSU:SIG:RFST:0.0200T/m')
        # m.groups()[0]
        if parser is float:
            return(float(re.findall("[-+]?\d*\.\d+|\d+", msg)[0]))
        return msg

    def _float_parser(self, msg):
        pass

    def _determine_magnet_axes(self):
        cat = self.ask('READ:SYS:CAT')
        self.axes = re.findall('DEV:GRP(.+?):PSU', cat)

    def _determine_current_to_field(self):
        # This has a unit A/T
        self._ATOB = {}
        for ax in self.axes:
            r = self._get_cmd('READ:DEV:GRP{}:PSU:ATOB?'.format(ax), float)
            self._ATOB[ax] = r

    def write(self, msg):
        rep = self.ask(msg)
        self._latest_response = rep
        if 'INVALID' in rep:
            raise Warning(rep)
