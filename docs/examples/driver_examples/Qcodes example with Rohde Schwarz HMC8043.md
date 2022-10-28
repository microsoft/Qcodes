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

# QCoDeS Example with R&S HMC 8043 Power Supply

```{code-cell} ipython3
import qcodes as qc
import qcodes.instrument_drivers.rohde_schwarz.HMC8043 as hmc8043
```

Create the instrument (in this case a HMC8043 connected with ethernet to the 10.0.1.1 address)

```{code-cell} ipython3
ps = hmc8043.RohdeSchwarzHMC8043('ps-1', 'TCPIP0::10.0.1.1::inst0::INSTR')
```

You can set voltage and/or current to any channel

```{code-cell} ipython3
ps.ch1.set_voltage(1)
ps.ch1.set_current(0.2)
ps.ch2.set_voltage(10)
```

Channel(s) should be turned on, as well as the master on/off

```{code-cell} ipython3
ps.ch1.state('ON')
ps.state('ON')
```

Voltage, current and power can be measured

```{code-cell} ipython3
print('V1=', ps.ch1.voltage())
print('I1=', ps.ch1.current())
print('P1=', ps.ch1.power())
```

And finally turned off

```{code-cell} ipython3
ps.ch1.state('OFF')
ps.state('OFF')
```

```{code-cell} ipython3

```
