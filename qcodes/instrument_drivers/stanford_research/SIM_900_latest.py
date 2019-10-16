import numpy as np
from typing import cast, Dict, Union
import time as time

from qcodes import VisaInstrument, InstrumentChannel, ParameterWithSetpoints, Parameter
from qcodes.utils.validators import Enum, Numbers
from qcodes.utils.helpers import create_on_off_val_mapping

# ------------------ SIM_900 CHASSIS ------------------
'''
Sim900 driver
Adopted to qcodes by: Zhang zhongming

This is the driver for the SIM 900 mainframe
Modules SIM 910 , SIM 928 and SIM 965 are written as submodules. 
Usage:
Initialize with
<name> = SIM_900('<name>', address='<GBIP address>', reset=<bool>)
e.g.
sim900 = SIM_900('sim900',address = 'GPIB::19')
To reference a submodule and one of it's parameters.
sim900._SIM_928_4.output.set(1)
'''
class SIM_900(VisaInstrument):
    def __init__(self, name: str, address: 'GPIB address', reset=False):
        super().__init__(name, address)
        '''
        Creates submodules objects with SIM_900 has their parent to manage the instrument
        SIM_910 is assumed to be connected to channel 1
        SIM_965 is assumed to be connected to channel 8
        Channels 2 to 6 are assumed to house SIM_928
        reference a SIM 928 installed in channel 2 using _SIM_928_2,
        _SIM_928_4 for channel 4 and so on.  
        '''
        for chnl in [2,3,4,5,6]:
            self.add_submodule(f"_SIM_928_{chnl}",SIM_928(self,"sim_928",chnl))
        self.add_submodule("_SIM_910",SIM_910(self,"sim_910"))
        self.add_submodule("_SIM_965",SIM_965(self,"sim_965"))

        if reset:
            self.reset()

    def all_output(self,val):
        value_mapping = {1:"OPON" , 0:"OPOF"}
        self.write(f"BRDT {value_mapping[val]}")
    def reset(self):
        self.write('RST')
# --------- SIM_928 voltage source -------------------
class SIM_928(InstrumentChannel):
    def __init__(self, parent, name:str, channel_number):
        super().__init__(parent,name)
        self.Channel_number = channel_number
        self.add_parameter("output", set_cmd = self.do_set_output, post_delay = 0.02)
        self.add_parameter("source", set_cmd = self.do_set_source,
                           get_cmd=self.do_get_source, vals=Numbers(-20,20),
                           get_parser=float, units='V', inter_delay = 0.01,
                           step=0.02)
        self.add_parameter("battery",get_cmd = self.battery_status) 
# ------ BATTERY ------------------
    def battery_change(self):
        channel = self.Channel_number
        self.write("SNDT %d,'BCOR'" %channel)

    def battery_status(self):
        channel = self.Channel_number
        self.write("CONN %d,'xyzzy'" %(channel))
        time.sleep(0.05)
        self.write("OVSR?")
        self.write('OVSR?')
        time.sleep(0.05)
        reply = self.ask('BATS?')
        self.write('xyzzy')
        reply1 = reply.split(',')
        reply_map = {'1': 'in use', '2': 'recharging', '3': 'on standby'}
        print(f'Battery A is {reply_map[reply1[0]]}')
        print(f'Battery B is {reply_map[reply1[1]]}')

# ------ OUTPUT ------------------
    def do_set_output(self,val):
        channel = self.Channel_number
        if int(val) == 1:
            self.write("SNDT %d,'OPON'" %(channel))
            print('Switching Source%d on' %(channel))
        elif int(val) == 0:
            self.write("SNDT %d,'OPOF'" %(channel))
            print('Switching Source%d off' %(channel))
# ------- SOURCE ----------------
# We can removed the np.linspace step increment with qcodes parameter step. After it has been tested
    def do_set_source(self,val):
        val = float(val)
        channel = self.Channel_number
        self.write("CONN %d,'xyzzy'" %(channel))
        time.sleep(0.01)
        self.write('VOLT %f' %(val))
        time.sleep(0.01)
        self.write('xyzzy')

    def do_get_source(self):
        channel = self.Channel_number
        reply = None
        self.write("CONN %d,'xyzzy'" %(channel))
        time.sleep(0.05)
        reply = self.ask('VOLT?')
        time.sleep(0.05)
        self.write('xyzzy')
        return reply
# ------------------ SIM_910 jfet amplifier ----------------------
class SIM_910(InstrumentChannel):
    def __init__(self,parent : 'SIM 900', name : str, Channel_number=1):
        super().__init__(parent,name)
        self.Channel_number = Channel_number
        self.add_parameter("gain",
                           set_cmd = self.do_set_gain,
                           get_cmd = self.do_get_gain
                           , vals = Enum(1,2,5,10,20,50,100),
                           post_delay = 0.01)
        self.add_parameter("jcoup",set_cmd = self.do_set_jcoup,get_cmd = self.do_get_jcoup
                           ,vals = Enum(1,2), post_delay = 0.01)
        self.add_parameter("shield",set_cmd = self.do_set_shield,get_cmd = self.do_get_shield
                           ,vals = Enum(1,2), post_delay = 0.02)
        self.add_parameter("input", set_cmd = self.do_set_input,get_cmd = self.do_get_input,
                           vals = Enum(1,2,3), post_delay = 0.01)

# ---------- GAIN ----------------------
    def do_set_gain(self, val):
        self.write("SNDT 1,'GAIN %d'" % val)
        print('Setting gain to %s' % val)

    def do_get_gain(self):
        chnl = self.Channel_number
        self.write(f"CONN {chnl},'xyzzy'")
        reply = self.ask('GAIN?')
        self.write('xyzzy')
        self.write(f"CONN {chnl},'xyzzy'")
        reply = self.ask('GAIN?')
        self.write('xyzzy')
        return reply
# --------- JCOUP ----------------------
    def do_set_jcoup(self, val):
        chnl = self.Channel_number
        self.write(f"SNDT {chnl},'COUP %d'" % val)
        if val == 1:
            print('Setting JFET coupling to AC coupling')
        elif val == 2:
            print('Setting JFET coupling to DC coupling')

    def do_get_jcoup(self):
        chnl = self.Channel_number
        self.write(f"CONN {chnl},'xyzzy'")
        reply = self.ask('COUP?')
        self.write('xyzzy')
        self.write(f"CONN {chnl},'xyzzy'")
        reply = int(self.ask('COUP?'))
        self.write('xyzzy')
        if reply == 1:
            print('JFET AC coupling')
        elif reply == 2:
            print('JFET DC coupling')
            
# ------------- SHIELD ------------------
    def do_set_shield(self, val):
        chnl = self.Channel_number
        self.write(f"SNDT {chnl},'SHLD %d'" % val)
        print('Setting shield to %s' % val)


    def do_get_shield(self):
        chnl = self.Channel_number
        self.write(f"CONN {chnl},'xyzzy'")
        reply = self.ask('SHLD?')
        self.write('xyzzy')
        self.write(f"CONN {chnl},'xyzzy'")
        reply = int(self.ask('SHLD?'))
        self.write('xyzzy')
        if reply == 1:
            print('A & B input shields floated')
        elif reply == 2:
            print('Input shields tied to amplifier ground')

# ---------- INPUT ---------------------
    def do_set_input(self, val):
        chnl = self.Channel_number
        self.write(f"SNDT {chnl},'INPT %d'" % val)
        print('Setting input to %s' % val)

    def do_get_input(self):
        chnl = self.Channel_number
        self.write(f"CONN {chnl},'xyzzy'")
        reply = self.ask('INPT?')
        self.write('xyzzy')
        self.write(f"CONN {chnl},'xyzzy'")
        reply = int(self.ask('INPT?'))
        self.write('xyzzy')
        if reply == 1:
            print('A')
        elif reply == 2:
            print ('AB')
        elif reply == 3:
            print('Front end ground')

# --------------- SIM_965 filter ------------------
class SIM_965(InstrumentChannel):
    def __init__(self,parent : 'SIM 900', name : str, Channel_number=8):
        super().__init__(parent,name)
        self.Channel_number = Channel_number
        self.add_parameter("freq",set_cmd = self.do_set_freq,get_cmd = self.do_get_freq,
                           vals = Numbers(1,5.00E5),get_parser = float ,units = "Hz",
                           post_delay = 0.01)
        self.add_parameter("filter",set_cmd = self.do_set_filtype, get_cmd = self.do_get_filtype
                           ,vals = Enum(0,1), post_delay = 0.01)
        self.add_parameter("fcoup",set_cmd = self.do_set_fcoup,get_cmd = self.do_get_fcoup,
                           vals = Enum(0,1), post_delay = 0.01)
        self.add_parameter("Pass",set_cmd = self.do_set_pass, get_cmd = self.do_get_pass,
                           vals = Enum(0,1), post_delay = 0.01)
        self.add_parameter("slope",set_cmd = self.do_set_slope, get_cmd = self.do_get_slope,
                           vals = Enum(12,24,36,48) , units = "dB", post_delay = 0.01)

# ---------------- FREQ --------------------
    def do_set_freq(self, val):
        chnl = self.Channel_number
        self.write(f"SNDT {chnl},'FREQ %d'" % val)
        print('Setting cutoff frequency to %s' % val)

    def do_get_freq(self):
        chnl = self.Channel_number
        self.write(f"CONN {chnl},'xyzzy'")
        time.sleep(0.05)
        reply = self.ask('FREQ?')
        time.sleep(0.05)
        self.write('xyzzy')
        time.sleep(0.05)
        self.write(f"CONN {chnl},'xyzzy'")
        time.sleep(0.05)
        reply = self.ask('FREQ?')
        time.sleep(0.05)
        self.write('xyzzy')
        print(reply)
        return reply      

# --------- FILTER -------------------
    def do_set_filtype(self, val):
        chnl = self.Channel_number
        self.write(f"SNDT {chnl},'TYPE %d'" % val)
        if val == 0:
            print('Setting filter type to Butter')
        elif val == 1:
            print('Setting filter type to Bessel')

    def do_get_filtype(self):
        chnl = self.Channel_number
        self.write(f"CONN {chnl},'xyzzy'")
        reply = self.ask('TYPE?')
        self.write('xyzzy')
        self.write(f"CONN {chnl},'xyzzy'")
        reply = int(self.ask('TYPE?'))
        self.write('xyzzy')
        if reply == 0:
            print('Butter')
        elif reply == 1:
            print('Bessel')
# --------- COUPLING -----------------
    def do_set_fcoup(self, val):
        chnl = self.Channel_number
        self.write(f"SNDT {chnl},'COUP %d'" % val)
        if val == 0:
            print('Setting Filter coupling to DC coupling')
        elif val == 1:
            print('Setting Filter coupling to AC coupling')

    def do_get_fcoup(self):
        chnl = self.Channel_number
        self.write(f"CONN {chnl},'xyzzy'")
        reply = self.ask('COUP?')
        self.write('xyzzy')
        self.write(f"CONN {chnl},'xyzzy'")
        reply = int(self.ask('COUP?'))
        self.write('xyzzy')
        if reply == 0:
            print('Filter DC coupling')
        elif reply == 1:
            print('Filter AC coupling')       

# --------------- PASS --------------------
    def do_set_pass(self, val):
        chnl = self.Channel_number
        self.write(f"SNDT {chnl},'PASS %d'" % val)
        if val == 0:
            print('Setting filter to LOWPASS')
        elif val == 1:
            print('Setting filter to HIGHPASS')

    def do_get_pass(self):
        chnl = self.Channel_number
        self.write(f"CONN {chnl},'xyzzy'")
        reply = self.ask('PASS?')
        self.write('xyzzy')
        self.write(f"CONN {chnl},'xyzzy'")
        reply = int(self.ask('PASS?'))
        self.write('xyzzy')
        if reply == 0:
            print('Filter LOWPASS')
        elif reply == 1:
            print('Filter HIGHPASS')

# -------------- SLOPE --------------------
    def do_set_slope(self, val):
        chnl = self.Channel_number
        self.write(f"SNDT {chnl},'SLPE %d'" % val)

    def do_get_slope(self):
        chnl = self.Channel_number
        self.write(f"CONN {chnl},'xyzzy'")
        reply = self.ask('SLPE?')
        self.write('xyzzy')
        self.write(f"CONN {chnl},'xyzzy'")
        reply = int(self.ask('SLPE?'))
        self.write('xyzzy')
        print(reply)
