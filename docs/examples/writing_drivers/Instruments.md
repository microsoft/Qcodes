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

# Instrument

An instrument is first and most fundamental pillar of QCoDeS as it represent the hardware you would want to talk to, either to control your system, collect data, or both.

Instruments come in several flavors:

- Hardware: most instruments map one-to-one to a real piece of hardware; in these instances, the QCoDeS Instrument requires a driver or communication channel to the hardware.
- Simulation: for theoretical or computational work, an instrument may contain or connect to a model that has its own state and generates results in much the same way that an instrument returns measurements.
- Manual: If a real instrument has no computer interface (just physical switches and knobs), you may want a QCoDeS instrument for it just to maintain a record of how it was configured at any given time during the experiment, and perhaps to use those settings in calculations. In these cases it is of course up to the user to keep the hardware and software synchronized.


## Instrument responsibilities:

- Holding connections to hardware, be it VISA, some other communication protocol, or a specific DLL or lower-level driver.
- Creating parameters and methods to support desired functionalities. These objects may be used independently of the instrument, but they make use of the instrument's hardware connections.
- Describing their current state (“snapshot”) when queried. These are supplied as a JSON-compatible dictionary supported by a custom JSON encoder class qcodes.utils.NumpyJSONEncoder.

Instruments hold state of:

- The communication address, and in many cases the open communication channel.
- A list of references to parameters added to the instrument.

## Instruments can fail:
When a VisaInstrument has been instantiated before, particularly with TCPIP, sometimes it will complain “VI_ERROR_RSRC_NFOUND: Insufficient location information or the requested device or resource is not present in the system” and not allow you to open the instrument until either the hardware has been power cycled or the network cable disconnected and reconnected.

## Example notebooks about Instruments:

- [Writing instrument drivers](Creating-Instrument-Drivers.ipynb): This notebook features examples for developing instrument drivers, including the definition of channels and drivers.
- [Simulated PyVISA instruments](Creating-Simulated-PyVISA-Instruments.ipynb): This provides examples of writing simulated instruments for testing software.
- [Abstract instruments](abstract_instruments.ipynb): This example focuses on creating abstraction to enforce standardized interfaces.
