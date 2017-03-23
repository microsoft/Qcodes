# code for example notebook

#%% Load packages
import time
import numpy as np
import logging

import qcodes as qc
import qcodes.instrument_drivers.QuTech.TimeStamp

#%% Create dummy instruments and model

from toymodel import AModel, MockGates, MockMeter

# now create this "experiment"
model = AModel()
gates = MockGates('gates', model=model)
meter = MockMeter('meter', model=model)

station = qc.Station(gates, meter)
ts = qcodes.instrument_drivers.QuTech.TimeStamp.TimeStampInstrument(name='TimeStamp')

# could measure any number of things by adding arguments to this
station.set_measurement(ts.timestamp, meter.amplitude)

#%% Test measurement

station.measure()

#%% Define process function


def progress(data):
    ''' Simpe progress meter, should be integrated with either loop or data object '''
    data.sync()
    tt=data.arrays['TimeStamp_timestamp']
    vv=~np.isnan(tt)        
    ttx=tt[vv]
    if ttx.size==0:
        return 0, np.Inf
    t0=ttx[0]
    t1=ttx[-1]
    
    logging.info('t0 %f t1 %f' % (t0, t1))

    fraction = ttx.size / tt.shape[0]
    remaining = (t1-t0)*(1-fraction)/fraction
    return fraction, remaining


#%% Go!
    
data = qc.Loop(gates.chan0[-20:20:0.1], 0.1).run(location='testsweep', overwrite=True)

for ii in range(10):
    print('progress: fraction %.2f, %.1f seconds remaining' % progress(data) )
    time.sleep(5)

