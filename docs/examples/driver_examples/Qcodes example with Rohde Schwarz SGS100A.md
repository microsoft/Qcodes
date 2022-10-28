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

# QCoDeS Example with Rohde Schwarz SGS100A RF source

```{code-cell} ipython3
import qcodes as qc

from qcodes.instrument_drivers.rohde_schwarz.SGS100A import RohdeSchwarz_SGS100A
```

```{code-cell} ipython3
sgsa = RohdeSchwarz_SGS100A("SGSA100", "TCPIP0::10.0.100.124::inst0::INSTR")
```

```{code-cell} ipython3
sgsa.print_readable_snapshot(update=True)
```

```{code-cell} ipython3
# set a power and a frequency
sgsa.frequency(10e9)
sgsa.power(-5)
```

```{code-cell} ipython3
# start RF output
sgsa.status(True)
```

```{code-cell} ipython3
# stop RF outout
sgsa.status(False)
```
