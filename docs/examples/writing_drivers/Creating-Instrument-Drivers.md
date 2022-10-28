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

# Creating QCoDeS instrument drivers

```{code-cell} ipython3
# most of the drivers only need a couple of these... moved all up here for clarity below
from time import sleep, time
import numpy as np
import ctypes  # only for DLL-based instrument

import qcodes as qc

from qcodes.instrument import (
    Instrument,
    VisaInstrument,
    ManualParameter,
    MultiParameter,
    InstrumentChannel,
    InstrumentModule,
)
from qcodes.utils import validators as vals
```

## Base Classes

There are 3 available:
- `VisaInstrument` - for most instruments that communicate over a text channel (ethernet, GPIB, serial, USB...) that do not have a custom DLL or other driver to manage low-level commands.
- `IPInstrument` - a deprecated driver just for ethernet connections. Do not use this; use `VisaInstrument` instead.
- `Instrument` - superclass of both `VisaInstrument` and `IPInstrument`, use this if you do not communicate over a text channel, for example:
  - PCI cards with their own DLLs
  - Instruments with only manual controls.

If possible, please use a `VisaInstrument`, as this allows for the creation of a simulated instrument. (See the [Creating Simulated PyVISA Instruments](Creating-Simulated-PyVISA-Instruments.ipynb) notebook)

+++

## Parameters and Channels

Broadly speaking, a QCoDeS instrument driver is nothing but an object that holds a connection handle to the physical instrument and has some sub-objects that represent the state of the physical instrument. These sub-objects are the `Parameters`. Writing a driver basically boils down to adding a ton of `Parameters`.

### What's a Parameter?

A parameter represents a single value of a single feature of an instrument, e.g. the frequency of a function generator, the mode of a multimeter (resistance, current, or voltage), or the input impedance of an oscilloscope channel. Each `Parameter` can have the following attributes:

  * `name`, the name used internally by QCoDeS, e.g. 'input_impedance'
  * `instrument`, the instrument this parameter belongs to, if any.
  * `label`, the label to use for plotting this parameter
  * `unit`, the physical unit. ALWAYS use SI units if a unit is applicable
  * `set_cmd`, the command to set the parameter. Either a SCPI string with a single '{}', or a function taking one argument (see examples below)
  * `get_cmd`, the command to get the parameter. Follows the same scheme as `set_cmd`
  * `vals`, a validator (from `qcodes.utils.validators`) to reject invalid values before they are sent to the instrument. Since there is no standard for how an instrument responds to an out-of-bound value (e.g. a 10 kHz function generator receiving 12e9 for its frequency), meaning that the user can expect anything from silent failure to the instrument breaking or suddenly outputting random noise, it is MUCH better to catch invalid values in software. Therefore, please provide a validator if at all possible.
  * `val_mapping`, a dictionary mapping human-readable values like 'High Impedance' to the instrument's internal representation like '372'. Not always needed. If supplied, a validator is automatically constructed.
  * `max_val_age`: Max time (in seconds) to trust a value stored in cache. If the parameter has not been set or measured more recently than this, an additional measurement will be performed in order to update the cached value. If it is ``None``, this behavior is disabled. ``max_val_age`` should not be used for a parameter that does not have a get function.
  * `get_parser`, a parser of the raw return value. Since all VISA instruments return strings, but users usually want numbers, `int` and `float` are popular `get_parsers`
  * `docstring` A short string describing the function of the parameter

Golden rule: if a `Parameter` is settable, it must always accept its own output as input.

There are two different ways of adding parameters to instruments. They are almost equivalent but comes with some trade-offs. We will show both below.
You may either declare the parameter as an attribute directly on the instrument or add it via the via the `add_parameter` method on the instrument class.

Declaring a parameter as an attribute directly on the instrument enables Sphinx, IDEs such as VSCode and static tools such as Mypy to work more fluently with
the parameter than if it is created via `add_parameter` however you must take care to remember to pass `instrument=self` to the parameter such that the
parameter will know which instrument it belongs to.
Instrument.add_parameter is better suited for when you want to dynamically or programmatically add a parameter to an instrument. For historical reasons most
instruments currently use `add_parameter`.


### Functions

Similar to parameters QCoDeS instruments implement the concept of functions that can be added to the instrument via `add_function`. They are meant to implement simple actions on the instrument such as resetting it. However, the functions do not add any value over normal python methods in the driver Class and we are planning to eventually remove them from QCoDeS. **We therefore encourage any driver developer to not use function in any new driver**.

### What's an InstrumentModule, then?

An `InstrumentModule` is a submodule of the instrument holding `Parameter`s. It sometimes makes sense to group `Parameter`s, for instance when an oscilloscope has four identical input channels (see Keithley example below)
or when it makes sense to group a particular set of parameters into their own module (such as a trigger module containing trigger related settings)

`InstrumentChannel` is a subclass of `InstrumentModule` which behaves identically to `InstrumentModule` you should chose either one depending on if you are implementing a module or a channel. As a rule of thumb you should use `InstrumentChannel` for something that the instrument has more than one of.

+++

## Naming Instruments

We are aiming to organize drivers in QCoDeS in a consistent way for easy discovery.
Note that not all drivers in QCoDeS are currently named consistently.
However, we aim to gradually update all drivers to be named as outlined above and any new driver
should be named in the way outlined below.

The same rules should apply for QCoDeS-contrib-drivers with the exception that all drivers are stored in subfolders of the drivers folder.

### Naming the Instrument class
A driver for an instrument with model `Model` and from the vendor `Vendor` should be stored in the file:

```
qcodes\instrument_drivers\{Vendor}\{Vendor}_{Model}.py
```
using snake case with an underscore between the vendor and model name but starting the
vendor name with upper case.

The primary instrument class should be named as follows:
```
class {Vendor}{Model}
     ...
```
E.g Vendor followed by Model number in CamelCase.

Note that we use vendor names starting with upper case for both folders and file names.

It is also fine to use an acronym for instrument vendors when there are well established. E.g. drivers for `American Magnetics Inc.` instruments
may use the acronym `AMI` to refer to the vendor.

As an example the driver for the Weinschel 8320 should be stored in the file `qcodes\instrument_drivers\Weinschel\Weinschel_8320.py` and the
class named `Weinschel8320`

### Naming InstrumentModule classes

`InstrumentModule`s and `InstrumentChannel`s should be defined in the same file as the driver that they are part of. The classes should preferable be named such that it is clear
from the name which instrument it belongs to. E.g a hypothetical `InstrumentChannel` belonging to a Weinschel 8320 should be named `Weinschel8320Channel` or similar.


### Driver supporting multiple models.

Often instrument vendors supply multiple instruments in a family with very similar specs only different in limits such as voltage ranges or
highest supported frequency.


As an example have a look at the Keysight 344xxA series of digital multi meters. To implement drivers for such instruments it is preferable
to implement a private base class such as `_Keysight344xxA`. This class should be stored either in a `private` subfolder of the Vendor folder or
in a file starting with an underscore i.e. `_Keysight344xxA.py`. If possible, we prefer a format where `x` is used to signal the parts of the model numbers that
may change. Along with this class subclasses for each of the supported models should be implemented. These may either make small modifications to the baseclass as needed
or be empty subclasses if no modifications are needed.

E.g. subclasses of the Keysight 344xxA driver for the specific model `34410A` should be named as `Keysight34410A` and stored in `Keysight34410A.py`.


+++

## Logging
Every QCoDeS module should have its own logger that is named with the name of the module. So to create a logger put a line at the top of the module like this:
```
log = logging.getLogger(__name__)
```
Use this logger only to log messages that are not originating from an `Instrument` instance. For messages from within an instrument instance use the `log` member of the `Instrument` class, e.g
```
self.log.info(f"Could not connect at {address}")
```
This way the instrument name will be prepended to the log message and the log messages can be filtered according to the instrument they originate from. See the example notebook of the logger module for more info ([offline](../logging/logging_example.ipynb),[online](https://nbviewer.jupyter.org/github/QCoDeS/Qcodes/tree/master/docs/examples/logging/logging_example.ipynb)).

When creating a nested `Instrument`, like e.g. something like the `InstrumentChannel` class, that has a `_parent` property, make sure that this property gets set before calling the `super().__init__` method, so that the full name of the instrument gets resolved correctly for the logging.

```{raw-cell}
## VisaInstrument: Simple example
The Weinschel 8320 driver is about as basic a driver as you can get. It only defines one parameter, "attenuation". All the comments here are my additions to describe what's happening.
```

```{code-cell} ipython3
class Weinschel8320(VisaInstrument):
    """
    QCoDeS driver for the stepped attenuator
    Weinschel is formerly known as Aeroflex/Weinschel
    """

    # all instrument constructors should accept **kwargs and pass them on to
    # super().__init__
    def __init__(self, name, address, **kwargs):
        # supplying the terminator means you don't need to remove it from every response
        super().__init__(name, address, terminator="\r", **kwargs)

        self.attenuation = Parameter(
            "attenuation",
            unit="dB",
            # the value you set will be inserted in this command with
            # regular python string substitution. This instrument wants
            # an integer zero-padded to 2 digits. For robustness, don't
            # assume you'll get an integer input though - try to allow
            # floats (as opposed to {:0=2d})
            set_cmd="ATTN ALL {:02.0f}",
            get_cmd="ATTN? 1",
            # setting any attenuation other than 0, 2, ... 60 will error.
            vals=vals.Enum(*np.arange(0, 60.1, 2).tolist()),
            # the return value of get() is a string, but we want to
            # turn it into a (float) number
            get_parser=float,
            instrument=self,
        )
        """Control the attenuation"""
        # The docstring below the Parameter declaration makes Sphinx document the attribute and it is therefore
        # possible to see from the documentation that the instrument has this parameter. It is strongly encouraged to
        # add a short docstring like this.

        # it's a good idea to call connect_message at the end of your constructor.
        # this calls the 'IDN' parameter that the base Instrument class creates for
        # every instrument (you can override the `get_idn` method if it doesn't work
        # in the standard VISA form for your instrument) which serves two purposes:
        # 1) verifies that you are connected to the instrument
        # 2) gets the ID info so it will be included with metadata snapshots later.
        self.connect_message()


# instantiating and using this instrument (commented out because I can't actually do it!)
#
# from qcodes.instrument_drivers.weinschel.Weinschel_8320 import Weinschel8320
# weinschel = Weinschel8320('w8320_1', 'TCPIP0::172.20.2.212::inst0::INSTR')
# weinschel.attenuation(40)
```

## VisaInstrument: a more involved example

The Keithley 2600 sourcemeter driver uses two channels. The actual driver is quite long, so here we show an abridged version that has:

- A class defining a `Channel`. All the `Parameter`s of the `Channel` go here.
- A nifty way to look up the model number, allowing it to be a driver for many different Keithley models



```{code-cell} ipython3
class KeithleyChannel(InstrumentChannel):
    """
    Class to hold the two Keithley channels, i.e.
    SMUA and SMUB.
    """

    def __init__(self, parent: Instrument, name: str, channel: str) -> None:
        """
        Args:
            parent: The Instrument instance to which the channel is
                to be attached.
            name: The 'colloquial' name of the channel
            channel: The name used by the Keithley, i.e. either
                'smua' or 'smub'
        """

        if channel not in ["smua", "smub"]:
            raise ValueError('channel must be either "smub" or "smua"')

        super().__init__(parent, name)
        self.model = self._parent.model
        vranges = self._parent._vranges
        iranges = self._parent._iranges

        self.volt = Parameter(
            "volt",
            get_cmd=f"{channel}.measure.v()",
            get_parser=float,
            set_cmd=f"{channel}.source.levelv={{:.12f}}",
            # note that the set_cmd is either the following format string
            #'smua.source.levelv={:.12f}' or 'smub.source.levelv={:.12f}'
            # depending on the value of `channel`
            label="Voltage",
            unit="V",
            instrument=self,
        )

        self.curr = Parameter(
            "curr",
            get_cmd=f"{channel}.measure.i()",
            get_parser=float,
            set_cmd=f"{channel}.source.leveli={{:.12f}}",
            label="Current",
            unit="A",
            instrument=self,
        )

        self.mode = Parameter(
            "mode",
            get_cmd=f"{channel}.source.func",
            get_parser=float,
            set_cmd=f"{channel}.source.func={{:d}}",
            val_mapping={"current": 0, "voltage": 1},
            docstring="Selects the output source.",
            instrument=self,
        )

        self.output = Parameter(
            "output",
            get_cmd=f"{channel}.source.output",
            get_parser=float,
            set_cmd=f"{channel}.source.output={{:d}}",
            val_mapping={"on": 1, "off": 0},
            instrument=self,
        )

        self.nplc = Parameter(
            "nplc",
            label="Number of power line cycles",
            set_cmd=f"{channel}.measure.nplc={{:.4f}}",
            get_cmd=f"{channel}.measure.nplc",
            get_parser=float,
            vals=vals.Numbers(0.001, 25),
            instrument=self,
        )

        self.channel = channel


class Keithley2600(VisaInstrument):
    """
    This is the qcodes driver for the Keithley2600 Source-Meter series,
    tested with Keithley2614B
    """

    def __init__(self, name: str, address: str, **kwargs) -> None:
        """
        Args:
            name: Name to use internally in QCoDeS
            address: VISA ressource address
        """
        super().__init__(name, address, terminator="\n", **kwargs)

        model = self.ask("localnode.model")

        knownmodels = [
            "2601B",
            "2602B",
            "2604B",
            "2611B",
            "2612B",
            "2614B",
            "2635B",
            "2636B",
        ]
        if model not in knownmodels:
            kmstring = ("{}, " * (len(knownmodels) - 1)).format(*knownmodels[:-1])
            kmstring += "and {}.".format(knownmodels[-1])
            raise ValueError("Unknown model. Known model are: " + kmstring)

        # Add the channel to the instrument
        for ch in ["a", "b"]:
            ch_name = f"smu{ch}"
            channel = KeithleyChannel(self, ch_name, ch_name)
            self.add_submodule(ch_name, channel)

        # display parameter
        # Parameters NOT specific to a channel still belong on the Instrument object
        # In this case, the Parameter controls the text on the display
        self.display_settext = Parameter(
            "display_settext",
            set_cmd=self._display_settext,
            vals=vals.Strings(),
            instrument=self,
        )

        self.connect_message()
```

## VisaInstruments: Simulating the instrument

As mentioned above, drivers subclassing `VisaInstrument` have the nice property that they may be connected to a simulated version of the physical instrument. See the [Creating Simulated PyVISA Instruments](Creating-Simulated-PyVISA-Instruments.ipynb) notebook for more information. If you are writing a `VisaInstrument` driver, please consider spending 20 minutes to also add a simulated instrument and a test.

+++

## DLL-based instruments
The Alazar cards use their own DLL. C interfaces tend to need a lot of boilerplate, so I'm not going to include it all. The key is: use `Instrument` directly, load the DLL, and have parameters interact with it.

```{code-cell} ipython3
class AlazarTech_ATS(Instrument):
    dll_path = "C:\\WINDOWS\\System32\\ATSApi"

    def __init__(self, name, system_id=1, board_id=1, dll_path=None, **kwargs):
        super().__init__(name, **kwargs)

        # connect to the DLL
        self._ATS_dll = ctypes.cdll.LoadLibrary(dll_path or self.dll_path)

        self._handle = self._ATS_dll.AlazarGetBoardBySystemID(system_id, board_id)
        if not self._handle:
            raise Exception(
                f"AlazarTech_ATS not found at " f"system {system_id}, board {board_id}"
            )

        self.buffer_list = []

        # the Alazar driver includes its own parameter class to hold values
        # until later config is called, and warn if you try to read a value
        # that hasn't been sent to config.
        self.add_parameter(
            name="clock_source",
            parameter_class=AlazarParameter,
            label="Clock Source",
            unit=None,
            value="INTERNAL_CLOCK",
            byte_to_value_dict={
                1: "INTERNAL_CLOCK",
                4: "SLOW_EXTERNAL_CLOCK",
                5: "EXTERNAL_CLOCK_AC",
                7: "EXTERNAL_CLOCK_10MHz_REF",
            },
        )

        # etc...
```

It's very typical for DLL based instruments to only be supported on Windows. In such a driver care should be taken to ensure that the driver raises a clear error message if it is initialized on a different platform. This is typically best done by
by checking `sys.platform` as below. In this example we are using `ctypes.windll` to interact with the DLL. `windll` is only defined on on Windows.

QCoDeS is automatically typechecked with MyPy, this may give some complications for drivers that are not compatible with multiple OSes as there is no supported way to disabling the typecheck on a per platform basis for a specific submodule.
Specifically MyPy will correctly notice that `self.dll` does not exist on non Windows platforms unless we add the line `self.dll: Any = None` to the example below. By giving `self.dll` the type `Any` we effectively disable any typecheck related to `self.dll` on non Windows platforms which is exactly what we want. This works because MyPy knows how to interprete the `sys.platform` check and allows `self.dll` to have different types on different OSes.

```{code-cell} ipython3
class SomeDLLInstrument(Instrument):
    dll_path = "C:\\WINDOWS\\System32\\ATSApi"

    def __init__(self, name, dll_path=None, **kwargs):
        super().__init__(name, **kwargs)

        if sys.platform != "win32":
            self.dll: Any = None
            raise OSError("SomeDLLInsrument only works on Windows")
        else:
            self.dll = ctypes.windll.LoadLibrary(dll_path)

        # etc...
```

## Manual instruments
A totally manual instrument (like the ithaco 1211) will contain only `ManualParameter`s. Some instruments may have a mix of manual and standard parameters. Here we also define a new `CurrentParameter` class that uses the ithaco parameters to convert a measured voltage to a current. When subclassing a parameter class (`Parameter`, `MultiParameter`, ...), the functions for setting and getting should be called `get_raw` and `set_raw`, respectively.

```{code-cell} ipython3
class CurrentParameter(MultiParameter):
    """
    Current measurement via an Ithaco preamp and a measured voltage.

    To be used when you feed a current into the Ithaco, send the Ithaco's
    output voltage to a lockin or other voltage amplifier, and you have
    the voltage reading from that amplifier as a qcodes parameter.

    ``CurrentParameter.get()`` returns ``(voltage_raw, current)``

    Args:
        measured_param (Parameter): a gettable parameter returning the
            voltage read from the Ithaco output.

        c_amp_ins (Ithaco_1211): an Ithaco instance where you manually
            maintain the present settings of the real Ithaco amp.

        name (str): the name of the current output. Default 'curr'.
            Also used as the name of the whole parameter.
    """

    def __init__(self, measured_param, c_amp_ins, name="curr", **kwargs):
        p_name = measured_param.name

        p_label = getattr(measured_param, "label", None)
        p_unit = getattr(measured_param, "units", None)

        super().__init__(
            name=name,
            names=(p_name + "_raw", name),
            shapes=((), ()),
            labels=(p_label, "Current"),
            units=(p_unit, "A"),
            instrument=instrument,
            **kwargs,
        )

        self._measured_param = measured_param

    def get_raw(self):
        volt = self._measured_param.get()
        current = (
            self.instrument.sens.get() * self.instrument.sens_factor.get()
        ) * volt

        if self.instrument.invert.get():
            current *= -1

        value = (volt, current)
        return value


class Ithaco1211(Instrument):
    """
    This is the qcodes driver for the Ithaco 1211 Current-preamplifier.

    This is a virtual driver only and will not talk to your instrument.
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        # ManualParameter has an "initial_value" kwarg, but if you use this
        # you must be careful to check that it's correct before relying on it.
        # if you don't set initial_value, it will start out as None.
        self.add_parameter(
            "sens",
            parameter_class=ManualParameter,
            initial_value=1e-8,
            label="Sensitivity",
            units="A/V",
            vals=vals.Enum(1e-11, 1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-4, 1e-3),
        )

        self.add_parameter(
            "invert",
            parameter_class=ManualParameter,
            initial_value=True,
            label="Inverted output",
            vals=vals.Bool(),
        )

        self.add_parameter(
            "sens_factor",
            parameter_class=ManualParameter,
            initial_value=1,
            label="Sensitivity factor",
            units=None,
            vals=vals.Enum(0.1, 1, 10),
        )

        self.add_parameter(
            "suppression",
            parameter_class=ManualParameter,
            initial_value=1e-7,
            label="Suppression",
            units="A",
            vals=vals.Enum(1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-4, 1e-3),
        )

        self.add_parameter(
            "risetime",
            parameter_class=ManualParameter,
            initial_value=0.3,
            label="Rise Time",
            units="msec",
            vals=vals.Enum(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000),
        )

    def get_idn(self):
        return {
            "vendor": "Ithaco (DL Instruments)",
            "model": "1211",
            "serial": None,
            "firmware": None,
        }
```

## Custom Parameter classes

When you call:
```
self.add_parameter(name, **kwargs)
```
you create a `Parameter`. But with the `parameter_class` kwarg you can invoke any class you want:
```
self.add_parameter(name, parameter_class=OtherClass, **kwargs)
```

- `Parameter` handles most common instrument settings and measurements.
  - Accepts get and/or set commands as either strings for the instrument's `ask` and `write` methods, or functions/methods. The set and get commands may also be set to `False` and `None`. `False` corresponds to "no get/set method available" (example: the reading of a voltmeter is not settable, so we set `set_cmd=False`). `None` corresponds to a manually updated parameter (example: an instrument with no remote interface).
  - Has options for translating between instrument codes and more meaningful data values
  - Supports software-controlled ramping
- Any other parameter class may be used in `add_parameter`, if it accepts `name` and `instrument` as constructor kwargs. Generally these should subclasses of `Parameter`, `ParameterWithSetpoints`, `ArrayParameter`, or `MultiParameter`.

+++

`ParameterWithSetpoints` is specifically designed to handle the situations where the instrument returns an array of data with assosiated setpoints. An example of how to use it can be found in the notebook [Simple Example of ParameterWithSetpoints](../Parameters/Simple-Example-of-ParameterWithSetpoints.ipynb)

`ArrayParameter` is an older alternative that does the same thing. However, it is significantly less flexible and much harder to use correct but used in a significant number of drivers. **It is not recommended for any new driver.**

`MultiParameter` is designed to for the situation where multiple different types of data is captured from the same instrument command.

It is important that parameters subclass forwards the `name`, `label(s)`, `unit(s)` and `instrument` along with any unknown `**kwargs` to the superclasss.

+++

### On/Off parameters

Frequently, an instrument has parameters which can be expressed in terms of "something is on or off". Moreover, usually it is not easy to translate the lingo of the instrument to something that can have simply the value of `True` or `False` (which are typical in software). Even further, it may be difficult to find consensus between users on a convention: is it `on`/`off`, or `ON`/`OFF`, or python `True`/`False`, or `1`/`0`, or else?

This case becomes even more complex if the instrument's API (say, corresponding VISA command) uses unexpected values for such a parameter, for example, turning an output "on" corresponds to a VISA command `DEV:CH:BLOCK 0` which means "set blocking of the channel to 0 where 0 has the meaning of the boolean value False, and alltogether this command actually enables the output on this channel".

This results in inconsistency among instrument drivers where for some instrument, say, a `display` parameter has 'on'/'off' values for input, while for a different instrument a similar `display` parameter has `'ON'`/`'OFF'` values or `1`/`0`.

Note that this particular example of a `display` parameter is trivial because the ambiguity and inconsistency for "this kind" of parameters can be solved by having the name of the parameter be `display_enabled` and the allowed input values to be python `bool` `True`/`False`.

Anyway, when defining parameters where the solution does not come trivially, please, consider setting `val_mapping` of a parameter to the output of `create_on_off_val_mapping(on_val=<>, off_val=<>)` function from `qcodes.parameters` package. The function takes care of creating a `val_mapping` dictionary that maps given instrument-side values of `on_val` and `off_val` to `True`/`False`, `'ON'`/`'OFF'`, `'on'`/`'off'`, and other commonly used ones. Note that when getting a value of such a parameter, the user will not get `'ON'` or `'off'` or `'oFF'` - instead, `True`/`False` will be returned.

+++

## Dynamically adding and removing parameters

Sometimes when conditions change (for example, the mode of operation of the instrument is changed from current to voltage measurement) you want different parameters to be available.

To delete existing parameters:
```
del self.parameters[name_to_delete]
```
And to add more, do the same thing as you did initially:
```
self.add_parameter(new_name, **kwargs)
```

+++

## Handling interruption of measurements


+++

A QCoDeS driver should be prepared for interruptions of the measurement triggered by a KeyboardInterrupt from the enduser.
If an interrupt happens at an unfortunate time i.e. while communicating with the instrument or writing results of a measurement this may leave the program in an inconsistent state e.g. with a command in the output buffer of a VISA instrument. To protect against this QCoDeS ships with a context manager that intercepts KeyBoardInterrupts and delays them until it is safe to stop the program. By default QCoDeS protects writing to the database and communicating with VISA instruments in this way.


However, there may be situations where a driver needs additional protection around a critical piece of code. The following example shows how a critical piece of code can be protected. The reader is encouraged to experiment with this using the `interrupt the kernel` button in this notebook. Note how the first KeyBoardInterrupt triggers a message to the screen and then executes the code within the context manager but not the code outside. Furthermore 2 KeyBoardInterrupts rapidly after each other will trigger an immediate interrupt that does not complete the code within the context manager. The context manager can therefore be wrapped around any piece of code that the end user should not normally be allowed to interrupt.

```{code-cell} ipython3
from qcodes.utils.delaykeyboardinterrupt import DelayedKeyboardInterrupt
import time

with DelayedKeyboardInterrupt():
    for i in range(10):
        time.sleep(0.2)
        print(i)
    print("Loop completed")
print("Executing code after context manager")
```

## Organization

Your drivers do not need to be part of QCoDeS in order to use them with QCoDeS, but we strongly encourage you to contribute them to the [qcodes contrib drivers](https://github.com/QCoDeS/Qcodes_contrib_drivers) project. That way we prevent duplication of effort, and you will likely get help making the driver better, with more features and better code.

Make one driver per module, inside a directory named for the company (or institution), within the `instrument_drivers` directory, following the convention:

`instrument_drivers.<company>.<model>.<company>_<model>`
- example: `instrument_drivers.AlazarTech.ATS9870.AlazarTech_ATS9870`

Although the class name can be just the model if it is globally unambiguous. For example:
- example: `instrument_drivers.stanford_research.SR560.SR560`

And note that due to mergers, some drivers may not be in the folder you expect:
- example: `instrument_drivers.tektronix.Keithley_2600.Keithley_2600_Channels`

+++

## Documentation

A driver should be documented in the following ways.

* All methods of the driver class should be documented including the arguments and return type of the function. QCoDeS docstrings uses the [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
* Parameters should have a meaningful docstring if the usage of the parameter is not obvious.
* An IPython notebook that documents the usage of the instrument should be added to `docs/example/driver_examples/Qcodes example with <company> <model>.ipynb` Note that we execute notebooks by default as part of the docs build. That is usually not possible for instrument examples so we want to disable the execution. This can be done as described [here](https://nbsphinx.readthedocs.io/en/latest/never-execute.html) editing the notebooks metadata accessible via `Edit/Edit Notebook Metadata` from the notebook interface.
