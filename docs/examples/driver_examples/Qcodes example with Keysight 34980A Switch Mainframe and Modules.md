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

# QCoDeS Example with Keysight 34980A Multifunction Switch / Measure Mainframe and Modules

+++

## Instrument Short info
The 34980A Multifunction Switch / Measure Unit is a compact, economical, one-box solution for medium to high-density switch/measure applications. It requires different types of modules, such as Matrix Modules, Multiplexer Modules, and RF & Microwave Modules, for different applications. In this example notebook, I'll forcus on the Matrix Modules.

There are four different Matrix Modules, 34931A, 34932A, 34933A, and 34934A. Each can be configured into different layout, such as 4x8, 8x32 matrix. In this example notebook, two 34934A modules are connected to slot 1 and slot 2 of the 34980A system.

Each 34934A module is configured into a 8x64 matrix (8 rows, 64 columns), and connected with each other by rows. As a result, they form a 8x128 matrix (8 rows, 128 columns).

```{code-cell} ipython3
from qcodes.instrument_drivers.Keysight.keysight_34980a import Keysight34980A
from pyvisa.errors import VisaIOError
from IPython.display import display, Markdown
```

```{code-cell} ipython3
# instantiate the Swith Matrix
try:
    switch_matrix = Keysight34980A("swmx", "TCPIP0::1.1.1.1::inst0::INSTR")
    print('using physical instrument')
except (ValueError, VisaIOError):
    # Below is an example how to set up the mock instrument
    import qcodes.instrument.sims as sims
    visalib = sims.__file__.replace('__init__.py', 'keysight_34980A.yaml@sim')

    switch_matrix = Keysight34980A('switch_matrix_sim',
                                   address='GPIB::1::INSTR',
                                   visalib=visalib)
    display(Markdown("""**Note: using simulated instrument.**

Due to limitations in pyvisa-sim the behavior will be different. See the keysight_34980A.yaml file for more detial."""))
```

```{code-cell} ipython3
# show the module information for each occupied slot
switch_matrix.system_slots_info
```

Each module is defined as an instance of the submodule class. To access the module in slot n, use the following:

**switch_matrix.module[n]**

where **n** is the **slot** number

```{code-cell} ipython3
# to access the module installed in slot 1:
switch_matrix.module[1]
```

```{code-cell} ipython3
# to access the module installed in slot 2:
switch_matrix.module[2]
```

It is also possible to access each module via the following:

```{code-cell} ipython3
# to access the module installed in slot 1:
switch_matrix.slot_1_34934A
```

```{code-cell} ipython3
# to access the module installed in slot 1:
switch_matrix.slot_2_34934A
```

## Basic Switch control, module level

+++

For each module, any of the 8 row can be connected to any of the 64 column.

```{code-cell} ipython3
switch_matrix.module[1].connect(2, 34)     # connect row 2 to column 34, , for module in slot 1
print(switch_matrix.module[1].is_closed(2, 34)) # check if row 2 is connected to column 34
```

```{code-cell} ipython3
switch_matrix.slot_1_34934A.disconnect(2, 34)  # disconnect row 2 from column 34, for module in slot 1
print(switch_matrix.module[1].is_open(2, 34))   # check if row 2 is disconnect from column 34
```

```{code-cell} ipython3
switch_matrix.module[2].connect(3, 45)     # connect row 3 to column 45, , for module in slot 2
print(switch_matrix.module[2].is_closed(3, 45)) # check if row 3 is connected to column 45
```

We expect the following command to raise an exception because the column value 67 is out of range for module in slot 1.

```{code-cell} ipython3
try:
    switch_matrix.module[1].connect(5, 67)
except ValueError as err:
    print(err)
```

Because the two 8x64 matrices are connected by row to form a 8x128 matrix, user should convert column 67 to the correct column in the second module: 67-64=3, then use the connect function of module in slot 2:

**switch_matrix.module_in_slot[2].connect_path(5, 3)**

```{code-cell} ipython3
switch_matrix.module[2].connect(5, 3)
print(switch_matrix.module[2].is_closed(5, 3))
```

To connect multiple channels: (again, the channels should belong to the one module)

```{code-cell} ipython3
connections = [(3, 14), (1, 59), (2, 6)]
switch_matrix.module[2].connect_paths(connections)
print(switch_matrix.module[2].are_closed(connections))
```

```{code-cell} ipython3
switch_matrix.module[2].disconnect_paths(connections)
print(switch_matrix.module[2].are_open(connections))
```

## About safety interlock

+++

The Safety Interlock feature prevents connections to the Analog Buses if no terminal block or properly-wired cable is connected to the module. When a module is in safety interlock state, all the channels are disconnected/open, user can not connect/close any channels.

In the event of safety interlock, there will be a warning message. In the following example I manually changed the flag, **DO NOT** do this in the real situation.

```{code-cell} ipython3
switch_matrix.module[2]._is_locked = True # DO NOT perform this action in real situation
```

```{code-cell} ipython3
switch_matrix.module[2].connect(1,2)
```

Actions which make not atempt to connect any channels are still work:

```{code-cell} ipython3
switch_matrix.module[2].is_open(3,4)
```

### For module 34934A only
There is a relay protection mode for 34934A module. The fastest switching speeds for relays in a given signal path are achieved using the FIXed or ISOlated modes, followed by the AUTO100 and AUTO0 modes. There may be a maximum of 200 Ohm of resistance, which can only be bypassed by "AUTO0" mode.
See user's guide and programmer's reference for detailed explanation.

```{code-cell} ipython3
# to see the current protection mode of the module in slot 1:
switch_matrix.module[1].protection_mode()
```

```{code-cell} ipython3
# to set the protection mode to 'AUTO0', for module in slot 1:
switch_matrix.module[1].protection_mode('AUTO0')
switch_matrix.module[1].protection_mode()
```

## Basic Switch control, instrument level

+++

We can disconnect all the channels in a certain module, or all the installed modules

```{code-cell} ipython3
# to disconnect the channels in module installed in slot 1
switch_matrix.disconnect_all(1)
```

```{code-cell} ipython3
# to disconnect the channels in all installed modules
switch_matrix.disconnect_all()
```

## So, when to use connect, connect_paths, is_closed, or are_closed?

+++

Yes, it seem very confusing because of the path/paths, and is/are.

The reason for the singular/plural is that the module can work with one row and column pair (r, c), or a list of row and column pairs [(r1, c1), (r2, c2)].

**When work with single row-column pair, use the singular version of functions**
 - instrument.module.connect(r, c)
 - instrument.module.is_closed(r,c)

**When work with list of row-column pairs, use the plural version of functions**
 - instrument.module.connect_paths([(r1, c1), (r2, c2)])
 - instrument.module.are_closed([(r1, c1), (r2, c2)])
