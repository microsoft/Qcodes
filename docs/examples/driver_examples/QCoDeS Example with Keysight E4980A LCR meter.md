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

# QCoDeS Example with Keysight E4980A LCR meter

+++

This example is following the "Capacitor Measurements" on P215 of the user's guide: https://literature.cdn.keysight.com/litweb/pdf/E4980-90230.pdf?id=789356

A 8400 pF (8E-9 F) leaded ceramic capacitor is connected to the LCR meter.

```{code-cell} ipython3
import numpy as np
import time
from qcodes.dataset import initialise_database, Measurement, new_experiment
from qcodes.dataset.plotting import plot_by_id

from qcodes.instrument_drivers.Keysight.keysight_e4980a import KeysightE4980A, E4980AMeasurements
```

```{code-cell} ipython3
meter = KeysightE4980A("lcr_e4980a", 'USB0::0x2A8D::0x2F01::MY46618801::INSTR')
```

```{code-cell} ipython3
meter.IDN()
```

```{code-cell} ipython3
meter.reset()
```

### Step 1. Set up the E4980A’s measurement conditions:

1. set frequency to be 1MHz (system default is 1kHz)

2. set the voltage level to be 1.5 V (system default is 1V)

```{code-cell} ipython3
meter.frequency(1e6)
freq = meter.frequency()
print(f'The frequency for normal measurement is set to be {freq/1e6} MHz.')
```

```{code-cell} ipython3
meter.voltage_level(1.5)
volt_lv = meter.voltage_level()
print(f'the voltage for measurement signal is set to be {volt_lv} V.')
```

### Step 2. (optional) Set up the corrections for the unit

In the "Capacitor Measurements" example, a Keysight 16047E Direct Couple Test Fixture (general purpose) was used. To compensate for its residuals and strays, an OPEN/SHORT correction is required.

However, for our setup with a leaded ceramic capacitor, this step may not be necessary.

```{code-cell} ipython3
meter.correction.open()
meter.correction.open_state('on')
meter.correction.short()
meter.correction.short_state('on')
```

### step 3. Set the meausurement function.

+++

User should chose one function from the follow list, for example "**E4980AMeasurements.CPD**".

    "CPD": "Capacitance - Dissipation factor",  (by default)
    "CPQ": "Capacitance - Quality factor",
    "CPG": "Capacitance - Conductance",
    "CPRP": "Capacitance - Resistance",
    "CSD": "Capacitance - Dissipation factor",
    "CSQ": "Capacitance - Quality factor",
    "CSRS": "Capacitance - Resistance",
    "LPD": "Inductance - Dissipation factor",
    "LPQ": "Inductance - Quality factor",
    "LPG": "Inductance - Conductance",
    "LPRP": "Inductance - Resistance",
    "LPRD": "Inductance - DC resistance",
    "LSD": "Inductance - Dissipation factor",
    "LSQ": "Inductance - Quality factor",
    "LSRS": "Inductance - Resistance",
    "LSRD": "Inductance - DC resistance",
    "RX": "Resistance - Reactance",
    "ZTD": "Absolute value of impedance - Theta in degree",
    "ZTR": "Absolute value of impedance - Theta in radiant",
    "GB": "Conductance - Susceptance",
    "YTD": "Absolute value of admittance - Theta in degree",
    "YTR": "Absolute value of admittance - Theta in radiant",
    "VDID": "DC voltage - DC current"

Note: 
1. CP vs CS: ***P*** means measured using parallel equivalent circuit model, and ***S*** means measured using series equivalent circuit model.
2. Same for LP and LS
3. RP vs RS: Equivalent ***p***arallel/***s***eries resistance

+++

By default, the measurement function is "CPD", which mean "Capacitance - Dissipation factor":

```{code-cell} ipython3
meter.measurement_function()
```

The "measurement" property will return a "MeasurementPair" class (sub class of "MultiParameter"):

```{code-cell} ipython3
measurement = meter.measurement
type(measurement)
```

which has the same name as the ***current*** "measurement_function":

```{code-cell} ipython3
measurement.name
```

User can view the parameters to be measured, and the corresponding units as following:

```{code-cell} ipython3
print(f'The parameters to be measured are {measurement.names}, with units {measurement.units}')
```

and take measurements by calling:

```{code-cell} ipython3
measurement()
```

User may also directly call the name of the physics parameters: (can't call the unit this way though)

```{code-cell} ipython3
print(f'The capacitance is measured to be {measurement.capacitance}')
print(f'The dissipation_factor is measured to be {measurement.dissipation_factor}')
```

To change to another measurement function, for example, "LPD" for Inductance measurement:

```{code-cell} ipython3
meter.measurement_function(E4980AMeasurements.LPD)
meter.measurement_function()
```

```{code-cell} ipython3
measurement = meter.measurement
print(f'The measurement "{measurement.name}" returns values {measurement.get()}')
print(f'for parameters {measurement.names}')
print(f'with units {measurement.units}')
```

Any "MeasurementPair" object, when first initialized, will dynamically generate the corresponding attributes. For the LPD measurement above, the "measurement" object will have attributes "inductance" instead of "capacitance":

```{code-cell} ipython3
measurement.inductance
```

```{code-cell} ipython3
try:
    measurement.capacitance
except AttributeError as err:
    print(err)
```

To validate the measurement, we can measure the impedance, and calculate the capacitance.

For a capacitor, we have Zc = - j / (2 * Pi * f * C), where Zc is the impedance of the capacitor, f is the frequency, and C is the capacitance.

+++

There are actually two methods to measure the impedance:
1. to use the "measurement_function" method as above, and choose "RX";
2. to use the "measure_impedance()" call, which will always return impedance in complex format, Z = R + iX, where Z is impedance, R is resistance, and X is the reactance.

(The results from the two methods should be the same.)

```{code-cell} ipython3
meter.measurement_function(E4980AMeasurements.RX)
measurement = meter.measurement
print(measurement.names)
print(measurement())
print(measurement.units)
```

```{code-cell} ipython3
imp = meter.measure_impedance
print(imp.names)
print(imp())
print(imp.units)
```

To calculate the impedance: ("imp" here is also a "MeasurementPair", so user can call the resistance/reactance directly.)

```{code-cell} ipython3
Zc = np.sqrt(imp.resistance**2 + imp.reactance**2)
print(f"The impedance is {Zc} Ohm.")
```

and the capacitance:

```{code-cell} ipython3
C = -1/(2*np.pi*freq*Zc)
print(f"The capacitance is {C}F.")
```

which is the same as what we got previously using the "CPD" function:

```{code-cell} ipython3
meter.measurement_function(E4980AMeasurements.CPD)
measurement = meter.measurement
print(f"The capacitance  is {measurement.capacitance} F.")
```

### To work with QCoDeS "Measurement":

```{code-cell} ipython3
initialise_database()
exp = new_experiment(
    name='capacitance_measurement',
    sample_name="no sample"
)
```

```{code-cell} ipython3
meas = Measurement()
meas.register_parameter(meter.frequency)
meas.register_parameter(meter.measurement, setpoints=(meter.frequency,))
```

```{code-cell} ipython3
with meas.run() as datasaver:
    for freq in np.linspace(1.0E6, 1.5E6, 5):
        meter.frequency(freq)
        time.sleep(1)
        value_pair = meter.measurement()
        datasaver.add_result((meter.frequency, freq),
                             (meter.measurement, value_pair))
    run_id = datasaver.run_id
```

```{code-cell} ipython3
plot = plot_by_id(run_id)
```

The dataset will have two items, one for "capacitance", and the other for "dissipation_factor":

```{code-cell} ipython3
dataset = datasaver.dataset
dataset.get_parameter_data()
```

To switch to another measurement function:

```{code-cell} ipython3
meter.measurement_function(E4980AMeasurements.RX)
meter.measurement_function()
```

which will measure the following:

```{code-cell} ipython3
E4980AMeasurements.RX.names
```

We need to re-set the measurement, and re-register parameters so that qcodes knows the correct units:

```{code-cell} ipython3
meas = Measurement()
meas.register_parameter(meter.frequency)
meas.register_parameter(meter.measurement, setpoints=(meter.frequency,))
```

```{code-cell} ipython3
with meas.run() as datasaver:
    for freq in np.linspace(1.0E6, 1.5E6, 5):
        meter.frequency(freq)
        time.sleep(1)
        value_pair = meter.measurement()
        datasaver.add_result((meter.frequency, freq),
                             (meter.measurement, value_pair))
    run_id = datasaver.run_id
```

```{code-cell} ipython3
plot = plot_by_id(run_id)
```

```{code-cell} ipython3
meter.reset()
```

```{code-cell} ipython3

```
