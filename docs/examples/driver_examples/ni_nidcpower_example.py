#!/usr/bin/python
#
# A test script for the national instruments nidcpower driver

from qcodes.instrument_drivers.national_instruments.nidcpower import NiSmu

# as a current source:
smu = NiSmu("Ni4139", channels = { 0 : 'I'})

smu.smu0.voltage_limit(12.0) # only settable for current sources
smu.smu0.current(1.0e-3)

print(smu.smu0.voltage())
print(smu.smu0.current())

smu.close()

# as a voltage source:
smu = NiSmu("Ni4139", channels = { 0 : 'V'})

smu.smu0.current_limit(10e-3) # only settable for voltage sources
smu.smu0.voltage(1.0)

print(smu.smu0.voltage())
print(smu.smu0.current())
print("in compliance: %s" % smu.smu0.in_compliance()) # should be true

# set an out of compliance output (test load is 5.5k)
smu.smu0.current_limit(5e-4)
smu.smu0.voltage(5.0)

print(smu.smu0.voltage())
print(smu.smu0.current())
print("in compliance: %s" % smu.smu0.in_compliance()) # should be false

smu.close()
