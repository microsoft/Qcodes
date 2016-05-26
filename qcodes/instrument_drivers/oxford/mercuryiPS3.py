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

from qcodes import IPInstrument
from qcodes.utils.validators import Strings, Enum, Anything


class MercuryiPS(IPInstrument):
    '''
    MercuryiPS Driver

    This is the qcodes driver for the Oxford MercuryiPS magnet power supply.

    Status: beta-version.
        TODO:
        - SAFETY!! we need to make sure the magnet is only ramped at certain
          conditions!
        - make ATOB a parameter, and move all possible to use _read_cmd, _write_cmd
        - Fix this call = ''
                   eval(call)
          stuff, I guess there is a smarter way of doing that?
        - this findall stuff in _get_cmd, is that smart?
    '''
    # def __init__(self, name, axes=None, **kwargs):
    #     super().__init__(name, terminator='\n', **kwargs)
    def __init__(self, name, address=None, port=None, axes=None, **kwargs):
        super().__init__(name, address=address, port=port, terminator='\n',
                         **kwargs)
        self.axes = axes
        self._ATOB = []
        self._latest_response = ''
        # for some reason the first call is always invalid?! need some kind of init?
        self.ask('*IDN?')

        if axes is None:
            self._determine_magnet_axes()
        self._determine_current_to_field()

        self.add_parameter('setpoint',
                           names=['B'+ax.lower()+'_setpoint' for ax in self.axes],
                           get_cmd=partial(self._get_fld, self.axes, 'FSET'),
                           set_cmd=partial(self._ramp_to_setpoint, self.axes, 'FSET'),
                           units=['T'for ax in self.axes],
                           vals=Anything())

        self.add_parameter('rate',
                           names=['rate_B'+ax.lower() for ax in self.axes],
                           get_cmd=partial(self._get_fld, self.axes, 'RFST'),
                           set_cmd=partial(self._ramp_to_setpoint, self.axes, 'RFST'),
                           units=['T/m'for ax in self.axes],
                           vals=Anything())

        self.add_parameter('fld',
                           names=['B'+ax.lower() for ax in self.axes],
                           get_cmd=partial(self._get_fld, self.axes, 'FLD'),
                           set_cmd=partial(self._ramp_to_setpoint, self.axes, 'FSET'),
                           units=['T'for ax in self.axes],
                           vals=Anything())

        self.add_parameter('fldC',
                           names=['B'+ax.lower() for ax in self.axes],
                           get_cmd=partial(self._get_fld,
                                           self.axes, 'CURR'),
                           set_cmd=partial(self._ramp_to_setpoint, self.axes, 'CSET'),
                           units=['T'for ax in self.axes],
                           vals=Anything())

        self.add_parameter('rtp',
                           names=['radius', 'theta', 'phi'],
                           get_cmd=partial(self._get_rtp,
                                           self.axes, 'FLD'),
                           set_cmd=partial(self._set_rtp, self.axes, 'FSET'),
                           units=['|B|', 'rad', 'rad'],
                           vals=Anything())

        self.add_parameter('rtpC',
                           names=['radius', 'theta', 'phi'],
                           get_cmd=partial(self._get_rtp,
                                           self.axes, 'CURR'),
                           set_cmd=partial(self._set_rtp, self.axes, 'CSET'),
                           units=['|B|', 'rad', 'rad'],
                           vals=Anything())

        # so we have radius, theta and phi in buffer
        self.rtp.get()

        self.add_parameter('radius',
                           get_cmd=self._get_r,
                           set_cmd=self._set_r,
                           units='|B|')
        self.add_parameter('theta',
                           get_cmd=self._get_theta,
                           set_cmd=self._set_theta,
                           units='rad')
        self.add_parameter('phi',
                           get_cmd=self._get_phi,
                           set_cmd=self._set_phi,
                           units='rad')

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
                               get_cmd=partial(self._get_fld,
                                               ax, 'CURR'),
                               set_cmd=partial(self._ramp_to_setpoint, ax, 'CSET'),
                               label='B'+ax.lower(),
                               units='T')
            self.add_parameter(ax.lower()+'_fld_wait',
                               get_cmd=partial(self._get_fld,
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
                               get_cmd=partial(self._get_fld, ax, 'CSET'),
                               set_cmd=partial(self._set_fld, ax, 'CSET'),
                               units='T')
            self.add_parameter(ax.lower()+'_rate',
                               get_cmd=partial(self._get_fld, ax, 'RFST'),
                               set_cmd=partial(self._set_fld, ax, 'RFST'),
                               units='T/m')
            self.add_parameter(ax.lower()+'_rateC',
                               get_cmd=partial(self._get_fld, ax, 'RCST'),
                               set_cmd=partial(self._set_fld, ax, 'RCST'),
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
        # if cmd is 'CSET':
        #     self._set_fld(ax, cmd, setpoint)
        # elif cmd is 'FSET':
        #     self._set_fld(ax, cmd, setpoint)
        self._set_fld(ax, cmd, setpoint)
        msg = ''
        # print(ax, cmd, setpoint)
        # time.sleep(1)
        # self.rtos()
        for axis in ax:
            msg = 'SET:DEV:GRP{}:PSU:ACTN:RTOS'.format(axis)
            self.write(msg)
        # self.ask('')

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
        if not isinstance(setpoint, list):
            setpoint = [setpoint]
        if cmd in ['CSET', 'RCST', 'CURR', 'PCUR', 'RCUR']:
            setpoint = np.array(self._ATOB) * np.array(setpoint)
            # print('a', setpoint)

        if len(ax) == 1:
            # print('b', self.axes.index(ax))
            setpoint = setpoint[self.axes.index(ax)]
            # print('c', setpoint)

        # print('d', ax, cmd, setpoint)
        msg = 'SET:DEV:GRP{}:PSU:SIG:{}:{:6f}'
        # print('e', msg)
        self._write_cmd(cmd, ax, setpoint, msg)

    def _get_fld(self, ax, cmd):
        # Could be FSET for setpoint
        #          FLD for field
        #          RFLD for rate
        #          PFLD persistent field reading
        fld = list(self._read_cmd(cmd, ax, float))

        if cmd in ['CSET', 'RCST', 'CURR', 'PCUR', 'RCUR']:
            fld = np.array(fld) / np.array(self._ATOB)

        # print(ax, cmd, fld)
        if len(ax) == 1:
            return fld[self.axes.index(ax)]
        return list(fld)

    def _get_rtp(self, ax, cmd):
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

    def _determine_magnet_axes(self):
        cat = self.ask('READ:SYS:CAT')
        self.axes = re.findall('DEV:GRP(.+?):PSU', cat)

    def _determine_current_to_field(self):
        # This has a unit A/T
        self._ATOB = []
        for ax in self.axes:
            r = self._get_cmd('READ:DEV:GRP{}:PSU:ATOB?'.format(ax), float)
            self._ATOB.append(r)

    def _read_cmd(self, cmd, axes, parser=None, fmt=None):
        fmt = fmt or 'READ:DEV:GRP{}:PSU:SIG:{}?'
        msg = ''
        msglist = []
        for axis in axes:
            msglist.append(fmt.format(axis, cmd))
        msg = '\n'.join(msglist)
        self._send(msg)
        rep = self._recv()
        data = [None] * len(axes)
        for i in range(20):
            for ln in rep.split('\n'):
                for ix, msg in enumerate(msglist):
                    if msg[5:-1] in ln:
                        val = ln.split(':')[-1]
                        if parser is float:
                            try:
                                val = float(re.findall("[-+]?\d*\.\d+|\d+", val)[0])
                            except:
                                continue
                        data[ix] = val
                    if not (None in data):
                        return data
            rep = self._recv()
        return data

    def _write_cmd(self, cmd, axes, setpoint, fmt=None, parser=None):
        fmt = fmt or 'SET:DEV:GRP{}:PSU:SIG:{}:{:4f}'
        msg = ''
        msglist = []
        if len(axes) == 1:
            setpoint = [setpoint]
        for ix, axis in enumerate(axes):
            msglist.append(fmt.format(axis, cmd, setpoint[ix]))
        msg = '\n'.join(msglist)
        self._send(msg)
        rep = self._recv()
        data = [None] * len(axes)
        for i in range(20):
            for ln in rep.split('\n'):
                for ix, msg in enumerate(msglist):
                    if msg[5:-1] in ln:
                        val = ln.split(':')[-1]
                        if parser is float:
                            try:
                                val = float(re.findall("[-+]?\d*\.\d+|\d+", val)[0])
                            except:
                                continue
                        data[ix] = val
                    if not (None in data):
                        return data
            rep = self._recv()
        # print(data)

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

    # def ask(self, msg):
    #     mc = msg.count(self._terminator)
    #     rep = super().ask(msg)
    #     for i in range(20):
    #         # print(rep)
    #         if not rep.startswith(':INVALID'):
    #             for n in range(mc):
    #                 rep2 = super().ask('')
    #                 if 'INVALID' in rep2:
    #                     break
    #                 rep += self._terminator
    #                 rep += rep2
    #             break
    #         rep = super().ask('')

    #     return rep

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
