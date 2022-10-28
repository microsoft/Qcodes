---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Abstract Instruments and parameters

Abstract parameters allow us to create abstract instrument types which are guaranteed to have certain parameters present. For instance, this will allow us to create a unified interface for all voltage sources.

Note: An instrument which contains abstract parameters shall be called an 'Abstract Instrument'

```{code-cell} ipython3
from qcodes import Instrument
```

```{code-cell} ipython3
class BaseVoltageSource(Instrument):
    """
    All abstract parameters *must* be implemented
    before this class can be initialized. This
    allows us to enforce an interface.
    """

    def __init__(self, name: str):
        super().__init__(name)

        self.add_parameter("voltage", unit="V", abstract=True)

        self.add_parameter("current", unit="A", get_cmd=None, set_cmd=None)
```

### We cannot instantiate a Instrument with abstract parameters.

```{code-cell} ipython3
try:
    bv = BaseVoltageSource("name")
except NotImplementedError as error:
    print(f"Error: {error}")
```

Instruments which fail to initialize are not registered:

```{code-cell} ipython3
BaseVoltageSource.instances()
```

### Units of parameters defined in sub classes *must* match units defined in the base class 

```{code-cell} ipython3
class WrongSource2(BaseVoltageSource):
    """
    We implement the voltage paramter with the wrong unit
    """

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        self.add_parameter("voltage", unit="mV")
```

```{code-cell} ipython3
try:
    WrongSource2("name4")
except ValueError as error:
    print(f"Error: {error}")
```

Instruments which fail to initialize due to the wrong unit are also not registered:

```{code-cell} ipython3
BaseVoltageSource.instances()
```

# Working subclass

```{code-cell} ipython3
class VoltageSource(BaseVoltageSource):
    """
    We implement the voltage paramter with the correct unit.
    Here we just implement it as a manual parameter but in a
    real instrument we would probably not do that.
    """

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        self.add_parameter("voltage", unit="V", set_cmd=None, get_cmd=None)
```

```{code-cell} ipython3
vs = VoltageSource("name")
```

```{code-cell} ipython3
vs.voltage(1)
```

```{code-cell} ipython3
vs.voltage()
```

This instrument is registered as expected.

```{code-cell} ipython3
VoltageSource.instances()
```
