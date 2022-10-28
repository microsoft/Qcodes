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

# ScaledParameter
Sometimes the values that we set/get on the computer are not the physical value that reach/originate from the sample. The ScaledParameter can be used to convert quantities with a simple linear relationship without offset.

```{code-cell} ipython3
from qcodes import ManualParameter, ScaledParameter
```

```{code-cell} ipython3
dac0 = ManualParameter('dac0', unit = 'V')
dac1 = ManualParameter('dac1', unit = 'V')
amplitude = ManualParameter('amplitude', initial_value=3.14, unit = 'V')
```

## Resistive voltage divider
The simplest case is a resistive divider, where the set voltage is divided by a fixed amount.

```{code-cell} ipython3
vd = ScaledParameter(dac0, division = 10)
```

```{code-cell} ipython3
vd(10)
```

```{code-cell} ipython3
print('Vd =',vd(), vd.unit,', real setted value =', dac0(), dac0.unit)
```

## Voltage multiplier
If the voltage is amplified, we can specify a `gain` value instead of `division`.

```{code-cell} ipython3
vb = ScaledParameter(dac1, gain = 30, name = 'Vb')
```

```{code-cell} ipython3
vb(5)
```

```{code-cell} ipython3
print('Vb =',vd(), vb.unit,', Original_value =', dac1(), dac1.unit)
```

## Transimpedance amplifier
The ScaledParameter can be used also for quantities that are read, like a current read by a transimpedance amplifier, digitized by a multimeter.
We can also specify a different unit from the wrapped parameter. The semantic of gain/division is inverted compared to the previous cases, since it is a value that we read.

```{code-cell} ipython3
Id = ScaledParameter(amplitude, division = 1e6, name = 'Id', unit = 'A')
```

```{code-cell} ipython3
print('Id =',Id(), Id.unit,', Read_value =', amplitude(), amplitude.unit)
```

The gain can be manually changed at any time

```{code-cell} ipython3
Id.division = 1e8
print('Id =',Id(), Id.unit,', Read_value =', amplitude(), amplitude.unit)
```

The gain/division can be itself a Qcodes paramter, for example if is a gain set by a remote instrument

```{code-cell} ipython3
remote_gain = ManualParameter('remote_gain', initial_value=1e6, unit = 'V/A')
```

```{code-cell} ipython3
Id.division = remote_gain
print('Id =',Id(), Id.unit,', Read_value =', amplitude(), amplitude.unit)
```

```{code-cell} ipython3
remote_gain(1e8)
print('Id =',Id(), Id.unit,', Read_value =', amplitude(), amplitude.unit)
```
