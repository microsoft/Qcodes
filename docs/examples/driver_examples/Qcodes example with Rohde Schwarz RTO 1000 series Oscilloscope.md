---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# QCoDeS Example with Rohde Schwarz RTO 1000 Series Oscilloscope

```{code-cell} ipython3
%matplotlib notebook
import matplotlib.pyplot as plt
import qcodes as qc
from qcodes.plots.qcmatplotlib import MatPlot
from qcodes.measure import Measure

from qcodes.instrument_drivers.rohde_schwarz.RTO1000 import RTO1000
```

```{code-cell} ipython3
# Select your scope based on its ip address
# rto = RTO1000('rto', 'TCPIP0::172.20.3.86::inst0::INSTR')
rto = RTO1000('rto', 'TCPIP0::192.168.0.3::inst0::INSTR', model='RTO1024')
```

```{code-cell} ipython3
# Before we do anything, let's make sure that the instrument is in a clean state
#
# WARNING: This resets the instrument. Only execute this cell if you are sure about it!

rto.reset()
```

```{code-cell} ipython3
# The display should be set to 'remote' for better performance regarding data acquisition
# Most of the time, however, we'd like to look at the oscilloscope to see data
# and therefore set it to 'view'

rto.display('view')
```

```{code-cell} ipython3
# To get an overview of all instrument settings, print_readable_sanpshot is great
rto.print_readable_snapshot(update=True)
```

## A signal to record

For this tutorial, we record a **100 kHz sine wave** with a peak-to-peak amplitude of **100 mV** and an offset of **10 mV**.

+++

## Input settings

Our waveform generator outputs 50 Ohm, so we better set the input coupling to DC 50 Ohm as well.
And turn on channel 1.

```{code-cell} ipython3
rto.ch1.coupling('DC')  # 'DC' means DC 50 Ohm, 'DCLimit' means DC 1 MOhm, 'AC' means AC 1 MOhm.
rto.ch1.state('ON')
```

## Triggering

Let us set up an edge trigger on channel 1 with a trigger level of 25 mV.

```{code-cell} ipython3
rto.trigger_source('CH1')
rto.trigger_type('EDGE')
rto.trigger_edge_slope('NEG')
rto.trigger_level(0.025)
```

## Acquisition pt. 1 (running continuously)

The oscilloscope can run continuously or acquire a fixed number of times (or of course not run at all).

```{code-cell} ipython3
rto.run_cont() # run continuously
```

## Horizontal settings

We want to see one period of the sine.

```{code-cell} ipython3
rto.timebase_range(1/100e3)
rto.timebase_position(5e-6)
```

## Vertical settings

We of course want the sine wave to fully occupy the oscilloscope screen.

```{code-cell} ipython3
# This is EXACTLY matching the waveform...
rto.ch1.range(0.1)
rto.ch1.offset(0)
```

```{code-cell} ipython3
#... but we increase it a bit due to noise
rto.ch1.range(0.11)
```

## Acquisition pt. 2 (averaging)

To average on the scope, you must activate the averaging arithmetics for
the relevant channel.

The averaging takes place irrespective of run mode. Here, we perform 1000 averages and then stop the acquisition by acquiring NxSINGLE shots.

If you average in continuous acquisition mode, you get a running average of the latest `num_acquisition` scope shots.

```{code-cell} ipython3
# Let's do 1000 averages of our sine wave

rto.num_acquisitions(1000)
rto.ch1.arithmetics('AVERAGE')  # other options: 'ENVELOPE' and 'OFF'
rto.run_single()  # and perform a single run.
```

```{code-cell} ipython3
# To avoid running average effects, we reset the aqcuisition to "normal" mode
rto.num_acquisitions(1)
rto.ch1.arithmetics('OFF')
rto.run_cont()
```

## Reading out the trace - RUN CONT

```{code-cell} ipython3
# Before making a serious recording, consider whether you want 8 bit or 16 bit resolution
# This option may not be available on all scopes

rto.high_definition_state('ON')  # 'ON' -> 16 bit, 'OFF' -> 8 bit
```

```{code-cell} ipython3
##########################
# A NOTE ABOUT BANDWIDTH #
##########################

# In non-high definition state, each channel has its own bandwidth, that may either be
# 'FULL', 'B800', 'B200', or 'B20', where the numbers correspond to a bandwidth limit
# in MHz. In high defition mode, a global bandwidth limit is used.
rto.high_definition_bandwidth(1e9)
```

```{code-cell} ipython3
# Next, the trace must be prepared. This ensures that all settings are correct
rto.ch1.trace.prepare_trace()
```

```{code-cell} ipython3
# Now make a measurement of the trace and display it
data_hd = Measure(rto.ch1.trace).run()
plot = MatPlot(data_hd.arrays['trace'])
```

```{code-cell} ipython3
# Compare this to the 8 bit version:
rto.high_definition_state('OFF')
rto.ch1.trace.prepare_trace()
data_ld = Measure(rto.ch1.trace).run()
plot = MatPlot(data_ld.arrays['trace'])
```

## Reading out the trace - RUN Nx SINGLE

Oftentimes it is desirable to read out an averaged trace.
The driver makes sure that the acqusition has completed before asking for the trace.

For this example, we apply white noise with a **100 mV** peak-to-peak amplitude and no offset.

```{code-cell} ipython3
# to see averaging effects, we make the time base long
rto.timebase_range(0.1)
```

```{code-cell} ipython3
rto.ch1.trace.prepare_trace()
```

```{code-cell} ipython3
rto.num_acquisitions(10)
rto.ch1.arithmetics('AVERAGE')  # other options: 'ENVELOPE' and 'OFF'
rto.run_single()  # and perform a single run. NOTE: YOU MUST CALL THIS AS THE LAST THING BEFORE GETTING THE TRACE!!!
```

```{code-cell} ipython3
data_avgd = Measure(rto.ch1.trace).run()
plot = MatPlot(data_avgd.arrays['trace'])
```

```{code-cell} ipython3
# We can see the averaging effect by increasing the number of averages

rto.num_acquisitions(500)
rto.ch1.arithmetics('AVERAGE')  # other options: 'ENVELOPE' and 'OFF'
rto.run_single()  # and perform a single run. NOTE: YOU MUST CALL THIS AS THE LAST THING BEFORE GETTING THE TRACE!!!
data_avgd = Measure(rto.ch1.trace).run()
plot = MatPlot(data_avgd.arrays['trace'])
```

## Measurements
Start 1 of the 8 mesurements

```{code-cell} ipython3
import time

def do_measurement(rto, nr_measurements, nr_samples):
    """
    Will clean prevous measurement en start a new one
    Args:
        rto: scope driver object
        nr_measurements: number of used/enabled measurements (up to 8)
        nr_samples: minimal number of samples before the measurement will stop
    """
    # Clear the prevous mesurement
    for meas_ch in range(1, nr_measurements + 1):
        rto.submodules['meas{}'.format(meas_ch)].clear()
    rto.opc()
    rto.run_continues()

    # Check if measurement is running
    if(rto.is_acquiring() == False):
        raise RuntimeError('Cannot start measuremet; scope is not trigged')

    # Wait for the scope to be ready
    while rto.submodules['meas{}'.format(nr_measurements)].event_count() < 10:
        time.sleep(0.1)
    rto.opc()

    ## Actual measurement
    for meas_ch in range(1, nr_measurements + 1):
        rto.submodules['meas{}'.format(meas_ch)].clear()
    while rto.submodules['meas{}'.format(nr_measurements)].event_count() < nr_samples:
        time.sleep(0.1)
    rto.stop_opc()
```

```{code-cell} ipython3
rto.timebase_range(1.5 * (1/100e3)) # Need a bit more than 1 period
rto.ch1.range(0.11) # Need a bit more the 100mV

# Configure measurement 1 to determine frequency
rto.meas1.enable("ON")
rto.meas1.source("C1W1") # See rto.meas1.sources for more options
rto.meas1.category("AMPTime") # See rto.meas1.categories for more options
rto.meas1.main("FREQuency")  # See rto.meas1.meas_type for more options
rto.meas1.clear()
rto.meas1.statistics_enable("ON")
rto.run_continues()

#start measurement
print("Wait for results")
do_measurement(rto, 1, 500)
actualFq = rto.meas1.result_avg() * 1e-3
print(f"Ch: 1; frequency: {actualFq:.0f}kHz")
```

```{code-cell} ipython3
print(rto.meas1.sources)
```
