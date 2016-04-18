# code for example notebook

#%% Load packages
import time
import numpy as np
import logging

import qcodes as qc
import qcodes.instrument_drivers.TimeStamp
#os.chdir('/home/eendebakpt/develop/Qcodes/docs/examples')

#%% Create dummy instruments

from toymodel import AModel, MockGates, MockSource, MockMeter

# now create this "experiment"
model = AModel()
gates = MockGates('gates', model)
source = MockSource('source', model)
meter = MockMeter('meter', model)

station = qc.Station(gates, source, meter)

c0, c1, c2, vsd = gates.chan0, gates.chan1, gates.chan2, source.amplitude

ts = qcodes.instrument_drivers.TimeStamp.TimeStampInstrument(name='TimeStamp')

# could measure any number of things by adding arguments to this
station.set_measurement(ts.timestamp, meter.amplitude)

#%% Test measurement

station.measure()

#%% Define process function


def progress(data):
    ''' Simpe progress meter, should be integrated with either loop or data object '''
    data.sync()
    tt=data.arrays['timestamp']
    vv=~np.isnan(tt)        
    ttx=tt[vv]
    t0=ttx[0]
    t1=ttx[-1]
    
    logging.info('t0 %f t1 %f' % (t0, t1))

    fraction = ttx.size / tt.size[0]
    remaining = (t1-t0)*(1-fraction)/fraction
    return fraction, remaining


#%% Go!
    
data = qc.Loop(c0[-20:20:0.1], 0.1).run(location='testsweep', overwrite=True)

for ii in range(10):
    print('progress: fraction %.2f, %.1f seconds remaining' % progress(data) )
    time.sleep(5)

