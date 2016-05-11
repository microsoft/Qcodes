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
import time
import numpy as np

from qcodes import VisaInstrument
from qcodes.utils.validators import Strings, Enum, Anything


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
    # def __init__(self, name, address=None, port=None, axes=None, **kwargs):
    #     super().__init__(name, address=address, port=port, terminator='\n',
    #                      **kwargs)
        self.axes = axes
        self._ATOB = {}
        self._latest_response = ''
        # for some reason the first call is always invalid?! need some kind of init?
        self.ask('*IDN?')
        IDN = self.ask('*IDN?')[4:]
        vendor, model, serial, firmware = map(str.strip, IDN.split(':'))

        self.IDN = {'vendor': vendor, 'model': model,
                    'serial': serial, 'firmware': firmware}
        return

        if axes is None:
            self._determine_magnet_axes()
        self._determine_current_to_field()

        self.add_parameter('setpoint',
                           get_cmd=partial(self._get_fld, self.axes, 'FSET'),
                           set_cmd=partial(self._ramp_to_setpoint, self.axes, 'FSET'),
                           labels=['B'+ax.lower() for ax in self.axes],
                           units=['T'for ax in self.axes],
                           vals=Anything())
        self.add_parameter('fld',
                           get_cmd=partial(self._get_fld, self.axes, 'FLD'),
                           set_cmd=partial(self._ramp_to_setpoint, self.axes, 'FSET'),
                           labels=['B'+ax.lower() for ax in self.axes],
                           units=['T'for ax in self.axes],
                           vals=Anything())

        self.add_parameter('fldC',
                           get_cmd=partial(self._get_fld_converted,
                                           self.axes, 'CURR'),
                           set_cmd=partial(self._ramp_to_setpoint, self.axes, 'CSET'),
                           labels=['B'+ax.lower() for ax in self.axes],
                           units=['T'for ax in self.axes],
                           vals=Anything())

        self.add_parameter('rtp',
                           get_cmd=partial(self._get_rtp,
                                           self.axes, 'FLD'),
                           set_cmd=partial(self._set_rtp, self.axes, 'FSET'),
                           labels=['radius', 'theta', 'phi'],
                           units=['|B|', 'rad', 'rad'],
                           vals=Anything())

        self.add_parameter('rtpC',
                           get_cmd=partial(self._get_rtp,
                                           self.axes, 'CURR'),
                           set_cmd=partial(self._set_rtp, self.axes, 'CSET'),
                           labels=['radius', 'theta', 'phi'],
                           units=['|B|', 'rad', 'rad'],
                           vals=Anything())

        # so we have radius, theta and phi in buffer
        self.rtp.get()

        self.add_parameter('radius',
                           get_cmd=self._get_r,
                           set_cmd=self._set_r,
                           label='radius',
                           unit='|B|')
        self.add_parameter('theta',
                           get_cmd=self._get_theta,
                           set_cmd=self._set_theta,
                           label='theta',
                           unit='rad')
        self.add_parameter('phi',
                           get_cmd=self._get_phi,
                           set_cmd=self._set_phi,
                           label='phi',
                           unit='rad')

        # self.add_parameter('ACTN',
        #                    get_cmd=self._ACTN,
        #                    set_cmd='SET:DEV:GRP{}:PSU:ACTN:'.format(ax)+'{}',
        #                    vals=Enum('HOLD', 'RTOS', 'RTOZ', 'CLMP'))
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
            self.add_parameter(ax.lower()+'_fld_wait',
                               get_cmd=partial(self._get_fld_converted,
                                               ax, 'CURR'),
                               set_cmd=partial(self._ramp_to_setpoint_and_wait, ax, 'CSET'),
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

    def rtos(self):
        for ax in self.axes:
            # How do I properly call those parameters from here?
            # self.{ax}_ACTN.set('HOLD')
            call = 'self.{}_ACTN.set("RTOS")'.format(ax.lower())
            eval(call)

    def to_zero(self):
        for ax in self.axes:
            # How do I properly call those parameters from here?
            # self.{ax}_ACTN.set('HOLD')
            call = 'self.{}_ACTN.set("RTOZ")'.format(ax.lower())
            eval(call)

    def _ACTN(self):
        actn = self._read_cmd('ACTN', self.axes,
                              fmt='READ:DEV:GRP{}:PSU:ACTN?')
        return actn

    def _ramp_to_setpoint(self, ax, cmd, setpoint):
        if cmd is 'CSET':
            self._set_fld_converted(ax, cmd, setpoint)
        elif cmd is 'FSET':
            self._set_fld(ax, cmd, setpoint)
        msg = ''
        # print(ax, cmd, setpoint)
        # time.sleep(1)
        # self.rtos()
        for axis in ax:
            msg = 'SET:DEV:GRP{}:PSU:ACTN:RTOS'.format(axis)
            self.write(msg)
        self.ask('')

    def _ramp_to_setpoint_and_wait(self, ax, cmd, setpoint):
        error = 0.2e-3
        fldc = getattr(self, ax.lower()+'_fldC')
        fldc.set(setpoint)
        # self._ramp_to_setpoint(ax, cmd, setpoint)

        while abs(fldc.get() - setpoint) > error:
            time.sleep(0.5)
        if setpoint == 0.0:
            # This ensures that the magnet wont try to go hold
            # some funny xe-5 value.
            fld = getattr(self, ax.lower()+'_fld')
            fld.set(setpoint)

    def _set_fld(self, ax, cmd, setpoint):
        # Could be FSET for setpoint
        #          RFST for rate
        msg = 'SET:DEV:GRP{}:PSU:SIG:{}:{:6f}'
        if isinstance(ax, list):
            msg2 = ''
            for i, axis in enumerate(ax):
                msg2 += msg.format(axis, cmd, setpoint[i])
                msg2 += self._terminator
                # self._set_fld_converted(axis, cmd, setpoint[i])
            self.write(msg2)
            return
        msg2 = msg.format(ax, cmd, setpoint)
        self.write(msg2)

    def _get_fld(self, ax, cmd):
        # Could be FSET for setpoint
        #          FLD for field
        #          RFLD for rate
        #          PFLD persistent field reading
        if isinstance(ax, list):
            return [self._get_fld(axis, cmd) for axis in ax]
        fld = self._get_cmd('READ:DEV:GRP{}:PSU:SIG:{}?'.format(ax, cmd), float)
        return fld

    def _set_fld_converted(self, ax, cmd, setpoint):
        # Could be CSET for setpoint
        #          RCST for rate
        #
        # We set current, not field, this gives higher resolution due to
        # limited number of digits
        msg = 'SET:DEV:GRP{}:PSU:SIG:{}:{:6f}'
        if isinstance(ax, list):
            msg2 = ''
            for i, axis in enumerate(ax):
                msg2 += msg.format(axis, cmd, setpoint[i] * self._ATOB[axis])
                msg2 += self._terminator
                # self._set_fld_converted(axis, cmd, setpoint[i])
            self.write(msg2)
            return
        cur = setpoint * self._ATOB[ax]
        msg2 = msg.format(ax, cmd, cur)
        self.write(msg2)

    def _get_fld_converted(self, ax, cmd):
        # Could be CSET for setpoint
        #          CURR for field
        #          RCUR for rate
        #          PCUR persistent current reading
        # We ask for current, not field, this gives higher resolution due to
        # limited number of digits
        # The conversion gives us a float with lots of digits, how to limit that?
        if isinstance(ax, list):
            fld = []
            for axis in ax:
                fld.append(self._get_fld_converted(axis, cmd))
                # print(axis, cmd, fld[-1])
            return fld
        curr = self._get_cmd('READ:DEV:GRP{}:PSU:SIG:{}?'.format(ax, cmd), float)
        return curr / self._ATOB[ax]

    def _get_rtp(self, ax, cmd):
        if cmd == 'CURR':
            fld = self._get_fld_converted(ax, cmd)
        elif cmd == 'FLD':
            fld = self._get_fld(ax, cmd)

        sphere = self._carttosphere(fld)
        self._radius, self._theta, self._phi = sphere
        return sphere

    def _set_rtp(self, ax, cmd, setpoint):
        fld = self._spheretocart(setpoint)
        self._ramp_to_setpoint(ax, cmd, fld)

    def _get_r(self):
        self.rtp.get()
        return self._radius

    def _set_r(self, val):
        self.rtp.set([val, self._theta, self._phi])

    def _get_theta(self):
        self.rtp.get()
        return self._theta

    def _set_theta(self, val):
        self.rtp.set([self._radius, val, self._phi])

    def _get_phi(self):
        self.rtp.get()
        return self._phi

    def _set_phi(self, val):
        self.rtpC.set([self._radius, self._theta, val])

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

    def _read_cmd(self, cmd, axes, parser=None, fmt='READ:DEV:GRP{}:PSU:SIG:{}?'):
        msg = ''
        msglist = []
        for axis in axes:
            msglist.append(fmt.format(axis, cmd))
        msg = '\n'.join(msglist)

        rep = self.ask(msg)
        data = [None] * len(axes)
        for i in range(20):
            for ln in rep.split('\n'):
                for ix, msg in enumerate(msglist):
                    if msg[5:-1] in rep:
                        val = ln.split(':')[-1]
                        if parser is float:
                            try:
                                val = float(re.findall("[-+]?\d*\.\d+|\d+", val)[0])
                            except:
                                # print(msg)
                                return None
                        data[ix] = val
                    if all(data):
                        break
                if all(data):
                    break
            if all(data):
                break
            rep = self.ask('')

        return data

    def _write_cmd(self, cmd, axes, setpoint, fmt='SET:DEV:GRP{}:PSU:SIG:{}:{:6f}', parser=None):
        msg = ''
        msglist = []
        for ix, axis in enumerate(axes):
            msglist.append(fmt.format(axis, cmd, setpoint[ix]))
        msg = '\n'.join(msglist)
        rep = self.ask(msg)
        data = [None] * len(axes)
        for i in range(20):
            print(rep)
            for ln in rep.split('\n'):
                for ix, msg in enumerate(msglist):
                    if msg[-1] in rep:
                        val = ln.split(':')[-1]
                        if parser is float:
                            try:
                                val = float(re.findall("[-+]?\d*\.\d+|\d+", val)[0])
                            except:
                                # print(msg)
                                return None
                        data[ix] = val
                    if all(data):
                        break
                if all(data):
                    break
            if all(data):
                break
            rep = self.ask('')
        print(data)
        # return data



    def _get_cmd(self, question, parser=None):
        # print(question)
        rep = self.ask(question)
        # print(rep)
        # print()
        self._latest_response = rep
        msg = rep[len(question):]
        # How would one macth this without specifying the units?
        # m = re.match('STAT:DEV:GRPX:PSU:SIG:RFST:(.+?)T/m',
        #              'STAT:DEV:GRPX:PSU:SIG:RFST:0.0200T/m')
        # m.groups()[0]
        if parser is float:
            try:
                return(float(re.findall("[-+]?\d*\.\d+|\d+", msg)[0]))
            except:
                # print(msg)
                return None
        return msg.strip()

    def write(self, msg):
        rep = self.ask(msg)
        self._latest_response = rep
        if 'INVALID' in rep:
            print('warning', msg, rep)

    def ask(self, msg):
        mc = msg.count(self._terminator)
        rep = super().ask(msg)
        for i in range(20):
            # print(rep)
            if not rep.startswith(':INVALID'):
                for n in range(mc):
                    rep2 = super().ask('')
                    if 'INVALID' in rep2:
                        break
                    rep += self._terminator
                    rep += rep2
                break
            rep = super().ask('')

        return rep

    def _spheretocart(self, sphere):
        """
        r,  theta,  phi = sphere
        """
        r,  theta,  phi = sphere
        x = (r * np.sin(theta) * np.cos(phi))
        y = (r * np.sin(theta) * np.sin(phi))
        z = (r * np.cos(theta))
        return [x,  y,  z]

    def _carttosphere(self, field):
        field = np.array(field)
        r = np.sqrt(np.sum(field**2))
        if r == 0:
            theta = 0
            phi = 0
        else:
            theta = np.arccos(field[2] / r);
            phi = np.arctan2(field[1],  field[0])
        return [r, theta, phi]
