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

# QCoDeS Example with Tektronix Keithley 7510 Multimeter

In this example we will show how to use a few basic functions of the Keithley 7510 DMM. We attached the 1k Ohm resistor to the front terminals, with no source current or voltage.

For more detail about the 7510 DMM, please see the User's Manual: https://www.tek.com/digital-multimeter/high-resolution-digital-multimeters-manual/model-dmm7510-75-digit-graphical-sam-0, or Reference Manual: https://www.tek.com/digital-multimeter/high-resolution-digital-multimeters-manual-9

```{code-cell} ipython3
from qcodes.instrument_drivers.tektronix.keithley_7510 import Keithley7510
```

```{code-cell} ipython3
dmm = Keithley7510("dmm_7510", 'USB0::0x05E6::0x7510::04450363::INSTR')
```

## To reset the system to default settings:

```{code-cell} ipython3
dmm.reset()
```

## To perform measurement with different sense functions:

+++

When first turned on, the default sense function is for DC voltage

```{code-cell} ipython3
dmm.sense.function()
```

to perform the measurement:

```{code-cell} ipython3
dmm.sense.voltage()
```

There'll be an error if try to call other functions, such as current:

```{code-cell} ipython3
try:
    dmm.sense.current()
except AttributeError as err:
    print(err)
```

To switch between functions, do the following:

```{code-cell} ipython3
dmm.sense.function('current')
dmm.sense.function()
```

```{code-cell} ipython3
dmm.sense.current()
```

And of course, once the sense function is changed to 'current', the user can't make voltage measurement

```{code-cell} ipython3
try:
    dmm.sense.voltage()
except AttributeError as err:
    print(err)
```

The available functions in the driver now are 'voltage', 'current', 'Avoltage', 'Acurrent', 'resistance', and 'Fresistance', where 'A' means 'AC', and 'F' means 'Four-wire'

```{code-cell} ipython3
try:
    dmm.sense.function('ac current')
except ValueError as err:
    print(err)
```

## To set measurement range (positive full-scale measure range):

+++

By default, the auto range is on

```{code-cell} ipython3
dmm.sense.auto_range()
```

We can change it to 'off' as following

```{code-cell} ipython3
dmm.sense.auto_range(0)
dmm.sense.auto_range()
```

Note: this auto range setting is for the sense function at this moment, which is 'current'

```{code-cell} ipython3
dmm.sense.function()
```

If switch to another function, the auto range is still on, by default

```{code-cell} ipython3
dmm.sense.function('voltage')
dmm.sense.function()
```

```{code-cell} ipython3
dmm.sense.auto_range()
```

to change the range, use the following

```{code-cell} ipython3
dmm.sense.range(10)
```

```{code-cell} ipython3
dmm.sense.range()
```

This will also automatically turn off the auto range:

```{code-cell} ipython3
dmm.sense.auto_range()
```

the allowed range (upper limit) value is a set of discrete numbers, for example, 100mV, 1V, 10V, 100V, 100V. If a value other than those allowed values is input, the system will just use the "closest" one:

```{code-cell} ipython3
dmm.sense.range(150)
dmm.sense.range()
```

```{code-cell} ipython3
dmm.sense.range(105)
dmm.sense.range()
```

The driver will not give any error messages for the example above, but if the value is too large or too small, there'll be an error message:

```{code-cell} ipython3
try:
    dmm.sense.range(0.0001)
except ValueError as err:
    print(err)
```

## To set the NPLC (Number of Power Line Cycles) value for measurements:

+++

By default, the NPLC is 1 for each sense function

```{code-cell} ipython3
dmm.sense.nplc()
```

To set the NPLC value:

```{code-cell} ipython3
dmm.sense.nplc(.1)
dmm.sense.nplc()
```

Same as the 'range' variable, each sense function has its own NPLC value:

```{code-cell} ipython3
dmm.sense.function('resistance')
dmm.sense.function()
```

```{code-cell} ipython3
dmm.sense.nplc()
```

## To set the delay:

+++

By default, the auto delay is enabled. According to the guide, "When this is enabled, a delay is added after a range or function change to allow the instrument to settle." But it's unclear how much the delay is.

```{code-cell} ipython3
dmm.sense.auto_delay()
```

To turn off the auto delay:

```{code-cell} ipython3
dmm.sense.auto_delay(0)
dmm.sense.auto_delay()
```

To turn the auto delay back on:

```{code-cell} ipython3
dmm.sense.auto_delay(1)
dmm.sense.auto_delay()
```

There is also an "user_delay", but it is designed for rigger model, please see the user guide for detail.

+++

To set the user delay time:

First to set a user number to relate the delay time with: (default user number is empty, so an user number has to be set before setting the delay time)

```{code-cell} ipython3
dmm.sense.user_number(1)
dmm.sense.user_number()
```

By default, the user delay is 0s:

```{code-cell} ipython3
dmm.sense.user_delay()
```

Then to set the user delay as following:

```{code-cell} ipython3
dmm.sense.user_delay(0.1)
dmm.sense.user_delay()
```

The user delay is tied to user number:

```{code-cell} ipython3
dmm.sense.user_number(2)
dmm.sense.user_number()
```

```{code-cell} ipython3
dmm.sense.user_delay()
```

For the record, the auto delay here is still on:

```{code-cell} ipython3
dmm.sense.auto_delay()
```

## To set auto zero (automatic updates to the internal reference measurements):

+++

By default, the auto zero is on

```{code-cell} ipython3
dmm.sense.auto_zero()
```

To turn off auto zero:

```{code-cell} ipython3
dmm.sense.auto_zero(0)
dmm.sense.auto_zero()
```

The auto zero setting is also tied to each function, not universal:

```{code-cell} ipython3
dmm.sense.function('current')
dmm.sense.function()
```

```{code-cell} ipython3
dmm.sense.auto_zero()
```

There is way to ask the system to do auto zero once:

```{code-cell} ipython3
dmm.sense.auto_zero_once()
```

See P487 of the Reference Manual for how to use auto zero ONCE. Note: it's not funtion-dependent.

+++

## To set averaging filter for measurements, including average count, and filter type:

+++

By default, averaging is off:

```{code-cell} ipython3
dmm.sense.average()
```

To turn it on:

```{code-cell} ipython3
dmm.sense.average(1)
dmm.sense.average()
```

Default average count value is 10, **remember to turn average on**, or it will not work:

```{code-cell} ipython3
dmm.sense.average_count()
```

To change the average count:

```{code-cell} ipython3
dmm.sense.average_count(23)
dmm.sense.average_count()
```

The range for average count is 1 to 100:

```{code-cell} ipython3
try:
    dmm.sense.average_count(200)
except ValueError as err:
    print(err)
```

There are two average types, repeating (default) or moving filter:

```{code-cell} ipython3
dmm.sense.average_type()
```

To make changes:

```{code-cell} ipython3
dmm.sense.average_type('MOV')
dmm.sense.average_type()
```
