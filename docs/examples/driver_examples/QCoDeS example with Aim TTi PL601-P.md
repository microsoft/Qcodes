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

# QCoDeS example with Aim TTi PL601-P

+++

This is the example notebook that presents the basic features of the QCoDeS driver for the Aim TTi PL601-P programmable power supply. Aim TTi currently has six programmable power supply models. Among these, PL068-P, PL155-P, PL303-P and PL601-P have single output channel. The models PL303QMD-P and PL303QMT-P have dual and triple output channels, respectively. All programmable Aim TTi power supplies have the same remote control interface, therefore, the property calls in this notebook applies to all. Their names and the corresponding number of channels are implemented in the driver. Upon connection, if the instrument model is one of the listed above, driver shall automatically determine the output channel count.

+++

## Basic Operation

Let us first import QCoDeS and the driver for the power supply:

```{code-cell} ipython3
import qcodes as qc
from qcodes.instrument_drivers.AimTTi.AimTTi_PL601P_channels import AimTTi
```

Now, we create a station to hold our instrument:

```{code-cell} ipython3
station = qc.Station()
```

We finalize the initialization of the instrument by instantiation and then adding it to the station:

```{code-cell} ipython3
tti = AimTTi('aimtti', 'ASRL3::INSTR')
station.add_component(tti)
```

Let us, first, examine the properties of our instrument via its ``snapshot``:

```{code-cell} ipython3
tti.print_readable_snapshot()
```

The model PL601-P has a single channel that is named in the instrument as ``ch1``. As depicted in the snapshot, the default values of voltage and current outputs are ``5V`` and ``0.135A``, respectively. We can accsess and set these values to the desired ones by calling the corresponding parameter of the output channel:

```{code-cell} ipython3
tti.ch1.volt(3)
```

```{code-cell} ipython3
tti.ch1.volt()
```

```{code-cell} ipython3
tti.ch1.volt.unit
```

```{code-cell} ipython3
tti.ch1.volt.label
```

Similarly, for the current we have:

```{code-cell} ipython3
tti.ch1.curr(0.1)
```

```{code-cell} ipython3
tti.ch1.curr()
```

```{code-cell} ipython3
tti.ch1.curr.unit
```

```{code-cell} ipython3
tti.ch1.curr.label
```

the PL601-P has two current ranges for the output called ``Low (1mA-500mA)`` and ``High (1mA-1500mA)`` range, associated with the integers ``1`` and ``2``, respectively.

```{code-cell} ipython3
tti.ch1.curr_range()
```

```{code-cell} ipython3
tti.ch1.curr_range(1)
```

```{code-cell} ipython3
tti.ch1.curr_range()
```

Here, we note that the output must be switched off before changing the current range. This is automatically handled by the QCoDeS driver, and your present output state is preserved.

```{code-cell} ipython3
tti.ch1.output()
```

```{code-cell} ipython3
tti.ch1.output(True)
```

```{code-cell} ipython3
tti.ch1.output()
```

```{code-cell} ipython3
tti.ch1.curr_range(2)
```

```{code-cell} ipython3
tti.ch1.curr_range()
```

```{code-cell} ipython3
tti.ch1.output()
```

If you have a specifically designed set up for a particular measurement and would like to reuse it, it can be saved to the internal set-up store of the power supply. There are ten available slots specified by the intergers ``0-9``. To examine this functionality, let us get a snapshot of the current set-up:

```{code-cell} ipython3
tti.print_readable_snapshot()
```

We now save this configuretion to the slot number ``0`` via

```{code-cell} ipython3
tti.ch1.save_setup(0)
```

Now, let us change voltage and current values along with the current range:

```{code-cell} ipython3
tti.ch1.volt(5)
```

```{code-cell} ipython3
tti.ch1.curr(0.2)
```

```{code-cell} ipython3
tti.ch1.curr_range(1)
```

```{code-cell} ipython3
tti.print_readable_snapshot()
```

Indeed, our changes successfully took place. Now, to have our old set up back, all we need to do is to load slote ``0``:

```{code-cell} ipython3
tti.ch1.load_setup(0)
```

```{code-cell} ipython3
tti.print_readable_snapshot()
```

In some cases, a constant incremental increase (decrease) in voltage and current output may be needed. In particular, we may want to change the latter output values dynamically during a repeated process. This can be done by using the pre-defined voltage and current step sizes (in Volts and Ampere, respectively):

```{code-cell} ipython3
tti.ch1.volt_step_size()
```

```{code-cell} ipython3
tti.ch1.curr_step_size()
```

The values of the step sizes can be changed as usual:

```{code-cell} ipython3
tti.ch1.volt_step_size(0.1)
```

```{code-cell} ipython3
tti.ch1.volt_step_size()
```

```{code-cell} ipython3
tti.ch1.curr_step_size(0.01)
```

```{code-cell} ipython3
tti.ch1.curr_step_size()
```

We can, now, make incremental changes to the current and voltage outputs accordingly:

```{code-cell} ipython3
tti.ch1.increment_volt_by_step_size()
```

```{code-cell} ipython3
tti.ch1.volt()
```

```{code-cell} ipython3
tti.ch1.decrement_volt_by_step_size()
```

```{code-cell} ipython3
tti.ch1.volt()
```

Similarly, for the current output, we have:

```{code-cell} ipython3
tti.ch1.increment_curr_by_step_size()
```

```{code-cell} ipython3
tti.ch1.curr()
```

```{code-cell} ipython3
tti.ch1.decrement_curr_by_step_size()
```

```{code-cell} ipython3
tti.ch1.curr()
```

Note that, the step sizes reset to the default values after a power cycle. Therefore, if you wish to have definite step sizes for a specific purpose, we suggest you to save your set up (see above) before turning off the instrument.

Finally, the current meter averaging can be turned on and off remotely. As there is no remote query for the status, user should observe the "Meter Average" signal light. To turn it on, we simply write:

```{code-cell} ipython3
tti.ch1.set_damping(1)
```

Upon success, the signal light should be on, as well. To turn the average off, we write:

```{code-cell} ipython3
tti.ch1.set_damping(0)
```
