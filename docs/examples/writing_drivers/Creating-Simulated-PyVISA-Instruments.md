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

# Creating Simulated PyVISA Instruments

When developing stuff in a large codebase like QCoDeS, it is often uncanningly easy to submit a change that breaks stuff. Therefore, _continuous integration_ is performed in the form of automated tests that run before new code is allowed into the codebase. The many tests of QCoDeS can be found in `qcodes.tests`. 

But how about drivers? They constitute the majority of the codebase, but how can we test them? Wouldn't that require a physical copy each instrument to be present on the California server where we run our tests? It used to be so, but not anymore! For drivers utilising PyVISA (i.e. `VisaInstrument` drivers), we may create simulated instruments to which the drivers may connect.

## What?

This way, we may instantiate drivers and run simple tests on them. Tests like:

  * Can the driver even instantiate? This is very relevant when underlying APIs change.
  * Is the drivers (e.g.) "voltage-to-bytecode" converter working properly?

## Not!

It is not feasible to simulate any but the most trivial features of the instrument. Simulated instruments can not and should not perform tests like:

  * Do we wait sufficiently long for this oscilloscope's trace to be acquired?
  * Does our driver handle overlapping commands of this AWG correctly?
  
## How?

The basic scheme goes as follows:

  * Write a `.yaml` file for the simulated instrument. The instructions for that may be found here: https://pyvisa-sim.readthedocs.io/en/latest/ and specifically here: https://pyvisa-sim.readthedocs.io/en/latest/definitions.html#definitions
  * Then write a test for your instrument and put it in `qcodes/tests/drivers`. The file should have the name `test_<nameofyourdriver>.py`. 
  * Check that all is well by running `$ pytest test_<nameofyourdriver>.py`.
  
Below is an example.

+++

## Example: Weinschel8320

The Weinschel 8320 is a very simple driver.

```{code-cell} ipython3
from qcodes.instrument.visa import VisaInstrument
import qcodes.utils.validators as vals
import numpy as np


class Weinschel8320(VisaInstrument):
    """
    QCoDeS driver for the stepped attenuator
    Weinschel is formerly known as Aeroflex/Weinschel
    """

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\r', **kwargs)

        self.add_parameter('attenuation', unit='dB',
                           set_cmd='ATTN ALL {:02.0f}',
                           get_cmd='ATTN? 1',
                           vals=vals.Enum(*np.arange(0, 60.1, 2).tolist()),
                           get_parser=float)

        self.connect_message()
```

### The `.yaml` file

+++

The simplest `.yaml` file that is still useful, reads, in all its glory:

+++

```
spec: "1.0"
devices:
  device 1:
    eom:
      GPIB INSTR:
        q: "\r"  # MAKE SURE! that this matches the terminator of the driver!
        r: "\r"
    error: ERROR
    dialogues:
      - q: "*IDN?"
        r: "QCoDeS, Weinschel 8320 (Simulated), 1337, 0.0.01"
            

resources:  
  GPIB::1::INSTR:
    device: device 1
```

+++

Note that since no physical connection is made, it doesn't matter what interface we pretend to use (GPIB, USB, ethernet, serial, ...). As a convention, we always write GPIB in the `.yaml` files.

We save the above file as `qcodes/instrument/sims/Weinschel_8320.yaml`. This simulates an instrument with no settable parameter; only an `*IDN?` response. This is enough to instantiate the instrument.

Then we may connect to the simulated instrument.

```{code-cell} ipython3
import qcodes.instrument.sims as sims
# path to the .yaml file containing the simulated instrument
visalib = sims.__file__.replace('__init__.py', 'Weinschel_8320.yaml@sim')

wein_sim = Weinschel8320('wein_sim',
                          address='GPIB::1::INSTR',  # This matches the address in the .yaml file
                          visalib=visalib
                          )
```

### The test

+++

Now we can write a useful test!

```{code-cell} ipython3
import pytest
from qcodes.instrument_drivers.weinschel.Weinschel_8320 import Weinschel8320
import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'Weinschel_8320.yaml@sim')


# The following decorator makes the driver
# available to all the functions in this module
@pytest.fixture(scope='function')
def driver():
    wein_sim = Weinschel8320('wein_sim',
                              address='GPIB::1::65535::INSTR',  
                              visalib=visalib
                              )
    yield wein_sim
    
    wein_sim.close()
    
    
def test_init(driver):
    """
    Test that simple initialisation works
    """
    
    # There is not that much to do, really.
    # We can check that the IDN string reads back correctly
    
    idn_dict = driver.IDN()
    
    assert idn_dict['vendor'] == 'QCoDeS'
    
```

Save the test as `qcodes/tests/drivers/test_weinschel_8320.py`. 

+++

Open a command line/console/terminal, navigate to the `qcodes/tests/drivers/` folder and run
```
>> pytest test_weinschel_8320.py
```

This should give you an output similar to
```
========================================= 1 passed in 0.73 seconds ==========================================
```

+++

## Congratulations! That was it.

+++

## Bonus example: including parameters in the simulated instrument

+++

It is also possible to add queriable parameters to the `.yaml` file, but testing that you can read those back is of limited value. You should only add them if your driver needs them to instantiate, e.g. if it checks that some range or impedance is configured correctly on startup, or - more generally - if a part of your driver code that you'd like to test needs it to run.

For the sake of this example, let us add a test that the driver's parameter's validator will reject an attenuation of less than 0 dBm. Note that this concrete test is redundant, since we have separate tests for validators. It is, however, an excellent example to learn from.

First we update the `.yaml` file to contain a property matching the parameter.

+++

```
spec: "1.0"
devices:
  device 1:
    eom:
      GPIB INSTR:
        q: "\r"  # MAKE SURE! that this matches the terminator of the driver!
        r: "\r"
    error: ERROR
    dialogues:
      - q: "*IDN?"
        r: "QCoDeS, Weinschel 8320 (Simulated), 1337, 0.0.01"

    properties:

      attenuation:
        default: 0
        getter:
          q: "ATTN? 1"  # the set/get commands have to simply be copied over from the driver
          r: "{:02.0f}"
        setter:
          q: "ATTN ALL {:02.0f}"          

resources:  
  GPIB::1::INSTR:
    device: device 1
```

+++

Notice that we don't include the the
```r: OK```
as the response of setting a property. This is in contrast to what https://pyvisa-sim.readthedocs.io/en/latest/definitions.html#properties does. The response of a successful setting of a parameter will not return 'OK'.

+++

Next we update the test script.

```{code-cell} ipython3
import pytest
from qcodes.instrument_drivers.weinschel.Weinschel_8320 import Weinschel_8320
import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'Weinschel_8320.yaml@sim')


# The following decorator makes the driver
# available to all the functions in this module
@pytest.fixture(scope='function')
def driver():
    wein_sim = Weinschel8320('wein_sim',
                              address='GPIB::1::INSTR',  
                              visalib=visalib
                              )
    yield wein_sim
    
    wein_sim.close()
    
    
def test_init(driver):
    """
    Test that simple initialisation works
    """
    
    # There is not that much to do, really.
    # We can check that the IDN string reads back correctly
    
    idn_dict = driver.IDN()
    
    assert idn_dict['vendor'] == 'QCoDeS'
    
    
def test_attenuation_validation(driver):
    """
    Test that incorrect values are rejected
    """
    
    bad_values = [-1, 1, 1.5]
    
    for bv in bad_values:
        with pytest.raises(ValueError):
            driver.attenuation(bv)
    
```

Open a command line/console/terminal, navigate to the `qcodes/tests/drivers/` folder and run
```
>> pytest test_weinschel_8320.py
```

This should give you an output similar to
```
========================================= 2 passed in 0.73 seconds ==========================================
```

+++

## That's it!
