# Copyright (c) 2015 SpinCore Technologies, Inc.
# http://www.spincore.com
#
# This software is provided 'as-is', without any express or implied warranty. 
# In no event will the authors be held liable for any damages arising from the 
# use of this software.
#
# Permission is granted to anyone to use this software for any purpose, 
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
#
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software in a
# product, an acknowledgement in the product documentation would be appreciated
# but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.
#
# Modified by Mark Johnson

import ctypes
c_float_p = ctypes.POINTER(ctypes.c_float), # create a float pointer type

try:
        spinapi = ctypes.CDLL("spinapi64")
except:
	try:
		spinapi = ctypes.CDLL("spinapi")
	except:
		raise("Failed to load spinapi library.")
	
def enum(**enums):
    return type('Enum', (), enums)
		
ns = 1.0
us = 1000.0
ms = 1000000.0

MHz = 1.0
kHz = 0.001
Hz = 0.000001
		
#Instruction enum
Inst = enum(
	CONTINUE = 0,
	STOP = 1,
	LOOP = 2,
	END_LOOP = 3,
	JSR = 4,
	RTS = 5,
	BRANCH = 6,
	LONG_DELAY = 7,
	WAIT = 8,
	RTI = 9
)
spinapi.pb_get_version.restype = (ctypes.c_char_p)
spinapi.pb_get_error.restype = (ctypes.c_char_p)

spinapi.pb_count_boards.restype = (ctypes.c_int)

spinapi.pb_init.restype = (ctypes.c_int)

spinapi.pb_select_board.argtype = (ctypes.c_int)
spinapi.pb_select_board.restype = (ctypes.c_int)

spinapi.pb_set_debug.argtype = (ctypes.c_int)
spinapi.pb_set_debug.restype = (ctypes.c_int)

spinapi.pb_set_defaults.restype = (ctypes.c_int)

spinapi.pb_core_clock.argtype = (ctypes.c_double)
spinapi.pb_core_clock.restype = (ctypes.c_int)

spinapi.pb_write_register.argtype = (ctypes.c_int, ctypes.c_int)
spinapi.pb_write_register.restype = (ctypes.c_int)

spinapi.pb_start_programming.argtype = (ctypes.c_int)
spinapi.pb_start_programming.restype = (ctypes.c_int)

spinapi.pb_stop_programming.restype = (ctypes.c_int)

spinapi.pb_start.restype = (ctypes.c_int)
spinapi.pb_stop.restype = (ctypes.c_int)
spinapi.pb_reset.restype = (ctypes.c_int)
spinapi.pb_close.restype = (ctypes.c_int)

spinapi.pb_inst_dds2.argtype = (
	ctypes.c_int, #Frequency register DDS0
	ctypes.c_int, #Phase register DDS0
	ctypes.c_int, #Amplitude register DDS0
	ctypes.c_int, #Output enable DDS0
	ctypes.c_int, #Phase reset DDS0
	ctypes.c_int, #Frequency register DDS1
	ctypes.c_int, #Phase register DDS1
	ctypes.c_int, #Amplitude register DDS1
	ctypes.c_int, #Output enable DDS1,
	ctypes.c_int, #Phase reset DDS1,
	ctypes.c_int, #Flags
	ctypes.c_int, #inst
	ctypes.c_int, #inst data
	ctypes.c_double, #timing value (double)
)
spinapi.pb_inst_dds2.restype = (ctypes.c_int)

### DDS.h ###
spinapi.pb_set_freq.restype = (ctypes.c_int)
spinapi.pb_set_freq.argtype = (ctypes.c_double)

spinapi.pb_set_phase.restype = (ctypes.c_int)
spinapi.pb_set_phase.argtype = (ctypes.c_double)

spinapi.pb_set_shape_defaults.restype = (ctypes.c_int)

spinapi.pb_select_dds.restype = (ctypes.c_int)
spinapi.pb_select_dds.argtype = (ctypes.c_int)

spinapi.pb_dds_load.restype = (ctypes.c_int)
spinapi.pb_dds_load.argtype = (
        c_float_p, # *data
        ctypes.c_int # device number
)

spinapi.pb_dds_set_envelope_freq.restype = (ctypes.c_int)
spinapi.pb_dds_set_envelope_freq.argtype = (
        ctypes.c_float, # envelope frequency
        ctypes.c_int    # the frequency register
)

spinapi.pb_set_amp.restype = (ctypes.c_int)
spinapi.pb_set_amp.argtype = (
        ctypes.c_float, # amplitude 
        ctypes.c_int,   # address of amplitude register
)


def pb_get_version():
	"""Return library version as UTF-8 encoded string."""
	ret = spinapi.pb_get_version()
	return str(ctypes.c_char_p(ret).value.decode("utf-8"))

def pb_get_error():
	"""Return library error as UTF-8 encoded string."""
	ret = spinapi.pb_get_error()
	return str(ctypes.c_char_p(ret).value.decode("utf-8"))
	
def pb_count_boards():
	"""Return the number of boards detected in the system."""
	return spinapi.pb_count_boards()
	
def pb_init():
	"""Initialize currently selected board."""
	return spinapi.pb_init()
	
def pb_set_debug(debug):
	return spinapi.pb_set_debug(debug)
	
def pb_select_board(board_number):
	"""Select a specific board number"""
	return spinapi.pb_select_board(board_number)
	
def pb_set_defaults():
	"""Set board defaults. Must be called before using any other board functions."""
	return spinapi.pb_set_defaults()
	
def pb_core_clock(clock):
	return spinapi.pb_core_clock(ctypes.c_double(clock))
	
def pb_write_register(address, value):
	return spinapi.pb_write_register(address, value)
	
def pb_start_programming(target):
	return spinapi.pb_start_programming(target)

def pb_stop_programming():
	return spinapi.pb_stop_programming()
	
def pb_start():
	return spinapi.pb_start()
	
def pb_stop():
	return spinapi.pb_stop()
	
def pb_reset(): 
	return spinapi.pb_reset()
	
def pb_close():
	return spinapi.pb_close()
	
	
	
################################################################################
###                                 DDS.h                                    ###
################################################################################

def pb_set_freq(freq):
        return spinapi.pb_set_freq(ctypes.c_double(freq))

def pb_set_phase(phase):
        return spinapi.pb_set_phase(ctypes.c_double(phase))

def pb_inst_dds2(*args):
	t = list(args)
	#Argument 13 must be a double
	t[13] = ctypes.c_double(t[13])
	args = tuple(t)
	return spinapi.pb_inst_dds2(*args)


def pb_inst_dds2_shape(*args):
	t = list(args)
	# Final argument must be a double
	t[-1] = ctypes.c_double(t[-1])
	args = tuple(t)
        return spinapi.pb_inst_dds2_shape(*args)

def pb_set_shape_defaults():
        return spinapi.pb_set_shape_defaults()

def pb_select_dds(dds_num):
        return spinapi.pb_select_dds(dds_num)

def pb_dds_load(data, device):
        return spinapi.pb_dds_load(c_float_p(data), device)

def pb_dds_set_envelope_freq(freq, n):
        return spinapi.pb_dds_set_envelope_freq(ctypes.c_double(freq), n)

def pb_set_amp(amp, addr):
        return spinapi.pb_set_amp(ctypes.c_float(amp), addr)
