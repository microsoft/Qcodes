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

# QCoDeS Example with Rigol DP832 Power Supply

```{code-cell} ipython3
import qcodes as qc
import qcodes.instrument_drivers.rigol.DP832 as dp832
```

Create the instrument (in this case a Rigol DP832 connected with ethernet to the 10.0.0.102 address)

```{code-cell} ipython3
ps = dp832.RigolDP832('ps', 'TCPIP0::10.0.0.102::inst0::INSTR')
```

You can set voltage and/or current to any channel

```{code-cell} ipython3
ps.ch1.set_voltage(1)
ps.ch1.set_current(0.2)
ps.ch2.set_voltage(10)
ps.ch3.set_current(2)
```

Channel(s) should be turned on

```{code-cell} ipython3
ps.ch1.state('on')
```

Voltage, current and power can be measured

```{code-cell} ipython3
print('V1=', ps.ch1.voltage())
print('I1=', ps.ch1.current())
print('P1=', ps.ch1.power())
```

DP832 supports Over- Voltage (OVP) and Current (OCP) protections

```{code-cell} ipython3
ps.ch1.ovp_value(1.2)
ps.ch1.ocp_value(0.05)
ps.ch1.ovp_state('on')
ps.ch1.ocp_state('on')
```

Working mode can be probed 9Voltage/Current regulatde, or unregulated)

```{code-cell} ipython3
ps.ch1.mode()
```

```{code-cell} ipython3

```
