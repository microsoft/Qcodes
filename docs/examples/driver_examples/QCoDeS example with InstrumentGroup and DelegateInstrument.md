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

# Qcodes example with InstrumentGroup driver

This notebooks explains how to use the `InstrumentGroup` driver.

## About
The goal of the `InstrumentGroup` driver is to combine several instruments as submodules into one instrument. Typically, this is meant to be used with the `DelegateInstrument` driver. An example usage of this is to create an abstraction for devices on a chip.

## Usage
The way it's used is mainly by specifying an entry in the station YAML. For instance, to create a Chip that has one or more Devices on it that point to different source parameters. The example below shows three devices, each of which is initialised in one of the supported ways. Device1 has only DelegateParameters, while device2 and device3 have both DelegateParameters and channels added. Device3 adds its channels using a custom channel wrapper class.

```{code-cell} ipython3
%%writefile example.yaml

instruments:
  dac:
    type: qcodes.tests.instrument_mocks.MockDAC
    init:
      num_channels: 3

  lockin1:
    type: qcodes.tests.instrument_mocks.MockLockin

  lockin2:
    type: qcodes.tests.instrument_mocks.MockLockin

  MockChip_123:
    type: qcodes.instrument.delegate.InstrumentGroup
    init:
      submodules_type: qcodes.instrument.delegate.DelegateInstrument
      submodules:
        device1:
          parameters:
            gate:
              - dac.ch01.voltage
            source:
              - lockin1.frequency
              - lockin1.amplitude
              - lockin1.phase
              - lockin1.time_constant
            drain:
              - lockin1.X
              - lockin1.Y
        device2:
          parameters:
            readout:
              - lockin1.phase
          channels:
            gate_1: dac.ch01
        device3:
          parameters:
            readout:
              - lockin1.phase
          channels:
            type: qcodes.tests.instrument_mocks.MockCustomChannel
            gate_1:
              channel: dac.ch02
              current_valid_range: [-0.5, 0]
            gate_2:
              channel: dac.ch03
              current_valid_range: [-1, 0]

      set_initial_values_on_load: true
      initial_values:
        device1:
          gate.step: 5e-4
          gate.inter_delay: 12.5e-4
        device2:
          gate_1.voltage.post_delay: 0.01
        device3:
          gate_2.voltage.post_delay: 0.03
```

```{code-cell} ipython3
import qcodes as qc
```

```{code-cell} ipython3
station = qc.Station(config_file="example.yaml")
lockin1 = station.load_lockin1()
lockin2 = station.load_lockin2()
dac = station.load_dac()
chip = station.load_MockChip_123(station=station)
```

```{code-cell} ipython3
chip.device1.gate()
```

```{code-cell} ipython3
dac.ch01.voltage()
```

```{code-cell} ipython3
chip.device1.gate(1.0)
chip.device1.gate()
```

```{code-cell} ipython3
dac.ch01.voltage()
```

```{code-cell} ipython3
chip.device1.source()
```

```{code-cell} ipython3
chip.device1.drain()
```

Device with channels/gates:

```{code-cell} ipython3
chip.device2.gate_1
```

Setting voltages to a channel/gate of device2:

```{code-cell} ipython3
print(chip.device2.gate_1.voltage())
chip.device2.gate_1.voltage(-0.74)
print(chip.device2.gate_1.voltage())
```

Check initial values of device3, from which only gate_2.voltage.post_delay was set.

```{code-cell} ipython3
chip.device3.gate_1.voltage.post_delay
```

```{code-cell} ipython3
chip.device3.gate_2.voltage.post_delay
```
