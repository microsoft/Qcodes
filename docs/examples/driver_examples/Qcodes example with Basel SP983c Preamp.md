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

# Qcodes example with Basel SP983c Preamp and its Remote SP983a

+++

This notebook explains how the Basel SP983c Preamp works and shows the main features of its and its remote's QCoDeS driver.

+++

## Basel SP983c Preamp

+++

This preamp is a low-noise high-stability (LNHS) I to V converter which offers unique features such as a floating input and the possibility to apply an external offset voltage. It offers five decades of gain, from 10^5 up to 10^9 V/A, and an integrated low-pass filter with a cut-off from 30 Hz to 1 MHz.

+++

## Features of Qcodes Basel SP983c Preamp driver

+++

The driver has three parameters - gain, fcut and offset_voltage. 'gain' and 'fcut' parameters are mirroring the gain and cut-off frequency setup on the instrument and can be set to the values available on the physical instrument. It is users responsibility to set these parameters on the driver to the values matching to the values set on the physical instrument.

+++

Let's try it ...

+++

Make the necessary imports ...

```{code-cell} ipython3
from qcodes.instrument_drivers.basel.sp983c import SP983C
from qcodes.instrument_drivers.Keysight.Keysight_34465A_submodules import Keysight_34465A
import qcodes.instrument.sims as sims
```

```{code-cell} ipython3
preamp = SP983C("basel_preamp")
```

gain can be set as:

```{code-cell} ipython3
preamp.gain(1e07)
```

and recalled as:

```{code-cell} ipython3
preamp.gain()
```

cut-off frequency can be set as:

```{code-cell} ipython3
preamp.fcut(300)
```

and recalled as:

```{code-cell} ipython3
preamp.fcut()
```

```{code-cell} ipython3
preamp.close()
```

### How to setup an input offset voltage source for the Basel SP983c Preamp?

+++

'offset_voltage' parameter can be set with a source input offset voltage parameter. The range of input offset voltage is -10 to 10 Volts. This input offset voltage is used for offsetting the voltage by the preamp in range -0.1 to 1 Volts. Let's try it with a dummy source parameter.

+++

#### Create a source as input offset voltage for Basel preamp

```{code-cell} ipython3
VISALIB = sims.__file__.replace('__init__.py', 'Keysight_34465A.yaml@sim')
dmm = Keysight_34465A('kt_34465A_sim', address="GPIB::1::INSTR", visalib=VISALIB)
```

```{code-cell} ipython3
dmm.volt()
```

#### 1. Instantiate Basel preamp with the source input offset voltage parameter

```{code-cell} ipython3
preamp1 = SP983C("basel_preamp1", input_offset_voltage=dmm.volt)
```

```{code-cell} ipython3
preamp1.offset_voltage()
```

```{code-cell} ipython3
preamp1.close()
```

#### 2. Or, instantiate the preamp without source input offset voltage parameter and assign it later

```{code-cell} ipython3
preamp2 = SP983C("basel_preamp2")
```

```{code-cell} ipython3
preamp2.offset_voltage.source = dmm.volt
```

```{code-cell} ipython3
preamp2.offset_voltage()
```

```{code-cell} ipython3
preamp2.close()
```

#### Close source instrument

```{code-cell} ipython3
dmm.close()
```

## Basel Preamp Remote Control SP983a

+++

Remote Control SP983a connects with the Basel SP983c Preamp and provides user friendly computer interface for the preamp. While remote is connected to the basel preamp, knobs for gain and fcut on basel preamp needs to be set to remote position. When you are using remote with the basel preamp, QCoDeS provided driver for remote control can be used directly and it can be imported in your scripts as follows.

```{code-cell} ipython3
from qcodes.instrument_drivers.basel.sp983c_remote import SP983A
```

This driver replaces the Basel Preamp driver when remote SP983a in use. The remote's driver has the same parameters 'gain', 'fcut' and 'offset_voltage' which can be used in exactly the same way as mentioned above after initialization of the remote's driver as below.

```{code-cell} ipython3
preamp_remote = SP983A("remote", address="address of the remote instrument")
```

In addition, remote's driver provides another parameter to check the overload status. It can be used as follows.

```{code-cell} ipython3
preamp_remote.overload_status()
```
