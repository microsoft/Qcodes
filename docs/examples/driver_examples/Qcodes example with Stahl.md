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

# QCoDeS Example with the Stahl Bias Sources

This notebook provides an example to how to set voltages with the Stahl Bias Sources.

```{code-cell} ipython3
from qcodes.instrument_drivers.stahl import Stahl
```

```{code-cell} ipython3
stahl = Stahl("stahl", "ASRL3")
```

```{code-cell} ipython3
stahl.channel[4].voltage(2)
v = stahl.channel[4].voltage()
print(v)
```

```{code-cell} ipython3
stahl.channel[4].voltage(-2)
v = stahl.channel[4].voltage()
print(v)
```

```{code-cell} ipython3
stahl.channel[4].voltage(0)
v = stahl.channel[4].voltage()
print(v)
```

```{code-cell} ipython3
stahl.channel[0].current()
```

```{code-cell} ipython3
stahl.channel[0].current.unit
```

```{code-cell} ipython3
stahl.temperature()
```

```{code-cell} ipython3
stahl.channel[1].is_locked()
```

```{code-cell} ipython3
stahl.output_type
```

```{code-cell} ipython3

```
