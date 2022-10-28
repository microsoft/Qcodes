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

# QCoDeS Example with Keysight B2200 Series Femto Leakage Switch Matrix

+++

## Instrument Short info
The Keysight B2200 Switch Matrix has 14 rows and up to 48 columns. In the terminology of the Keysight B2200, the *Rows* are called *Inputs* and the *Columns* are called *Outputs*. In this example Notebook I will use the *Input/Output* terminology.

  **Note: Technically the Inputs could also be used as Outputs and vice versa.**

The inputs are subdivided into
  - 8 triaxial BNC input ports labeled "SMU Input" with number 1 to 8
  - 6 normal BNC input ports labeled "AUX Input" with number 9 to 14

The output ports are split into 4 modules with 12 ports each, hence a Switch matrix can support 12, 24, 36 or 48 output ports. The Instrument can either address the Modules individually (*Normal Config Mode*) or treat all installed modules as one (*Auto Config Mode*). In *Auto Config Mode* the output channels are numbered 1 to 48 (or whatever the maximum number of channels is if not all 4 modules are installed).

  **Note: The Qcodes driver only suports Auto Config Mode**

```{code-cell} ipython3
import qcodes as qc

from qcodes.instrument_drivers.Keysight.keysight_b220x import KeysightB220X
from pyvisa.errors import VisaIOError

from IPython.display import display, Markdown
```

```{code-cell} ipython3
# Create a station to hold all the instruments

station = qc.Station()

# instantiate the Switch Matrix and add it to the station
try:
    switch_matrix = KeysightB220X('switch_matrix',
                        address='TCPIP::192.168.111.100::gpib0,22::INSTR')
    print("using physical instrument")
except (ValueError, VisaIOError):
    # Either there is no VISA lib installed or there was no real instrument found at the
    # specified address => use simulated instrument
    import qcodes.instrument.sims as sims
    path_to_yaml = sims.__file__.replace('__init__.py', 'keysight_b220x.yaml')

    switch_matrix = KeysightB220X('switch_matrix',
                        address='GPIB::1::INSTR',
                        visalib=path_to_yaml + '@sim'
                        )
    display(Markdown("""**Note: using simulated instrument.**

Due to limitations in pyvisa-sim the behavior will be different. Especially
  - the default values are not recovered when calling `switch_matrix.reset()` method
  - commands will simply overwrite the current state (e.g. when calling `switch_matrix.gnd_enabled_inputs([3, 4])` and after that `switch_matrix.gnd_enabled_inputs([5, 6])`, the real instrument would return `[3,4,5,6]` when calling `switch_matrix.gnd_enabled_inputs()`, but the simulated instrument will only return `[5,6]`)"""))

station.add_component(switch_matrix)
```

## Basic Switch control

Any of the 14 Input ports can be connected to any of the 48 output ports

```{code-cell} ipython3
switch_matrix.reset()

print(switch_matrix.connections()) # {None}
```

```{code-cell} ipython3
switch_matrix.connect(2, 48) # connect input 2 to output 48

print(switch_matrix.connections()) # {(2, 48)}
```

```{code-cell} ipython3
switch_matrix.connect(14, 3) # connect input 14 to output 3

print(switch_matrix.connections()) # {(2, 48), (14, 3)}
```

```{code-cell} ipython3
switch_matrix.disconnect(14, 3) # disconnect input 14 from output 3

print(switch_matrix.connections()) # {(2, 48)}
```

```{code-cell} ipython3
switch_matrix.disconnect_all() # opens all switches

print(switch_matrix.connections()) # {None}
```

```{code-cell} ipython3
# To connect multiple input->output pairs it is faster
# to use the connect_paths method instead of calling connect in a loop
switch_matrix.connect_paths({(3, 45), (11, 9)})

print(switch_matrix.connections()) # {(11, 9), (3, 45)}
```

```{code-cell} ipython3
# Multiple switches can be disconnected with the disconnect_paths method
switch_matrix.disconnect_paths({(3, 45), (11, 9)})

print(switch_matrix.connections()) # {None}
```

## Advanced Switch Control
---

### Connection Rule
The Switch matrix has two *connection rules*:
  - in *free route* mode (**Default** after reset):
    - each input port can be connected to multiple output ports
    - and each output port can be connected to multiple input ports.
    - **Caution: If the Free connection rule has been specified, ensure multiple input ports are not connected to the same output port. Such configurations can cause damage**
  - in *single route* mode:
    - each input port can be connected to only one output port
    - and each output port can be connected to only one input port.
    - existing connection to a port will be disconnected when a new connection is made.
    
**Note: when switching from *free* to *single* mode, 1-to-multiple connections will not be changed until the connection is changed!!**

+++

#### Free Route Mode

```{code-cell} ipython3
switch_matrix.reset()

print(switch_matrix.connections()) # {None}
```

```{code-cell} ipython3
switch_matrix.connect(2, 33)

print(switch_matrix.connections()) # {(2, 33)}
```

```{code-cell} ipython3
switch_matrix.connect(2, 12) # Input 2 is connected to multiple outputs

print(switch_matrix.connections()) # {(2, 33), (2, 12)}
```

```{code-cell} ipython3
# ! DANGER ZONE !
# Connecting multiple inputs to the same output: This can cause damage if there
# two sources on the input pins as they will be short circuited
# 
# Uncomment and set to `True` if you know what you're doing:
# I_know_what_I_am_doing = False
# I_have_checked_there_are_no_two_sources_on_Input_5_and_6 = False
switch_matrix.reset()
print(switch_matrix.connections()) # {None}

try:
    assert I_know_what_I_am_doing
    assert I_have_checked_there_are_no_two_sources_on_Input_5_and_6
    
    switch_matrix.connect(5, 14)
    print(switch_matrix.connections()) # {(5, 14)}
    
    switch_matrix.connect(6, 14) # Input 5 and 6 are now both connected to Output 14
    print(switch_matrix.connections()) # {(5, 14), (6, 14)}

except (AssertionError, NameError):
    print("Skipped `Connecting multiple inputs to the same output` example.")
```

#### Single Route mode

```{code-cell} ipython3
switch_matrix.disconnect_all()
print(switch_matrix.connections()) # {None}

# Single route mode is activated like so:
switch_matrix.connection_rule('single')
```

```{code-cell} ipython3
switch_matrix.connect(2, 48)
print(switch_matrix.connections()) # {(2, 48)}

switch_matrix.connect(5, 9)
print(switch_matrix.connections()) # {(2, 48), (5, 9)}

# The following command will implicitly release the 2->48 connection because 
# input 2 will be used to connect to output 48
switch_matrix.connect(2, 12)
print(switch_matrix.connections()) # {(2, 12), (5, 9)}
```

**Note that even though a connection like {2->33, 2->12} or {5->14, 6->14} would be impossible 
to *create* in *Single Route* Mode it can still exist if it was created previously in *Free Route* mode!**

The reason is if the connection was made in *Free Route* mode and after that *Single Route* was activated the connections will **persist**. The following example illustrates this:

```{code-cell} ipython3
# Example:
switch_matrix.reset() # After reset *free route* mode is active
print(switch_matrix.connections()) # {None}

switch_matrix.connect(2,18) # Connections: {(2, 18)}
switch_matrix.connect(2,44) # Connections: {(2, 18), (2, 44)}
switch_matrix.connect(2,45) # Connections: {(2, 18), (2, 44), (2, 45)}

switch_matrix.connection_rule('single') # *Single Route* Mode is activated
# Still multiple Outputs connected to Input 2! 
print(switch_matrix.connections()) # Connections: {(2, 18), (2, 44), (2, 45)}

# The following command will only implcitly release the 2->45 connection:
switch_matrix.connect(5,45) # Still multiple Outputs connected to Input 2!
print(switch_matrix.connections()) # Connections: {(2, 18), (2, 44), (5, 45)}
```

---
### Connection Sequence
As explained above in **Single Route** Mode the switch matrix implicitly disconnects the previous path when a new connection is made.

There are options for how exactly this is done:
  - Break Before Make (`'bbm'`) (**Default** after reset):
    1. Disconnect previous route.
    2. Wait for relays to open.
    3. Connect new route.
  - Make Before Break (`'mbb'`):
    1. Connect new route.
    2. Wait for relays to close.
    3. Disconnect previous route.
  - No Sequence (`'none'`)
    1. Disconnect previous route.
    2. Connect new route.

```{code-cell} ipython3
switch_matrix.reset()

# After Reset the switch matrix is in *Free Route* Mode
# Although setting a Connection Sequence can also be done in *Free Route* mode,
# it **only has effect in Single Route Mode** (because in Free Route mode no connections are implicitely opened)
switch_matrix.connection_rule('single')

# *Break Before Make* is Default after Reset
print('Connection Sequence after Reset: ', switch_matrix.connection_sequence() )

# Change connection sequence to *Make Before Break*
switch_matrix.connection_sequence('mbb')

# Change connection sequence to *No Sequence*
switch_matrix.connection_sequence('none')
```

---
### Ground Mode
The Switch Matrix can automatically connect unused inputs and/or outputs to a designated Ground input. When the ground mode is *ON*, the input ground port is connected to
  - to *ground enabled* output ports that are not connected to any other input ports.
  - all inputs that are specified as *unused*

**Note: Connection sequence (to connect input ground port to output ports) is always Break-Before-Make.**

In principle the Ground Input Port can be chosen freely, however it is advisable to use the default ground input (AUX 12) as I will explain below.

**Note: The port that is selected as *Ground Input Port* cannot be used in normal `connect`/`disconnect` commands**

**Note: Ground mode cannot be set to ON when the bias mode is ON**

To make use of the Ground mode several things have to be done:
  - specify which *outputs* should be *ground enabled*
  - specify which *inputs* will be unused and should be *ground enabled*
  - (most often not necessary (see above)) specify which input port should be the ground input
  - Set Ground mode *ON*

The following will explain these steps

+++

#### Specifying which outputs should be *ground enabled*

There are two possibilities to specify which *outputs* should be *ground enabled*:

```{code-cell} ipython3
# Either only enable specific output ports
switch_matrix.gnd_enable_output(33)
switch_matrix.gnd_enable_output(11)

# Or Ground enable all output ports:
switch_matrix.gnd_enable_all_outputs()
```

#### Specifying which inputs are unused
Input ports that are marked as *unused* cannot be used in `connect`/`disconnect` commands

```{code-cell} ipython3
# Specify unused inputs with a list
switch_matrix.unused_inputs([5, 6, 7, 8])

print("Ground enabled inputs: ", switch_matrix.unused_inputs())
```

**Note: Everytime `unused_inputs(<list of unused inputs>)` is called, the previous setting is overwritten!**

```{code-cell} ipython3
print("Before: ", switch_matrix.unused_inputs())

switch_matrix.unused_inputs([3]) # when marking single inputs as unused they must also be wrapped in a list!

print("After: ", switch_matrix.unused_inputs())

# Calling `set_unused_input([3])` overwrote the previous setting ([5,6,7,8])
```

This means if you want to use all inputs again, simply call `unused_inputs()` with an empty list

```{code-cell} ipython3
switch_matrix.unused_inputs([])
print("Ground enabled inputs: ", switch_matrix.unused_inputs())
```

#### (Optional) Specifying which *input port* should be the *ground input*
In general designating an input port as *ground input* **does not automatically connect the inner conductor to GND**! So typically a BNC Short Circuit Connector Cap is required to connect the port to GND.

The **exception to this are input 12 and 13**, both have special internal circuitry. If *ground mode* is set to *ON* and
  - If Input 12 is selected as *ground input*, then the inner conductor of the BNC connector is **left floating** and the **internal inner conductor is connected to GND**
  - If Input 13 is selected as *ground input*, then the **inner conductor of the BNC connector AND the internal inner conductor is connected to GND**
  
So for both Input 12 and 13 no external Short Circuit Connector Cap is required. 

**Note: For Input 13 care must be taken that no source is connected to the input as otherwise it will be shorted to GND which might cause damage**.

```{code-cell} ipython3
# The default *ground input* port is input AUX 12.
print("The Ground Input Port is: ", switch_matrix.gnd_input_port())
```

```{code-cell} ipython3
# I_have_made_sure_that_there_is_no_source_connected_to_input_13 = False
try:
    assert I_have_made_sure_that_there_is_no_source_connected_to_input_13
    
    # Changing the ground input port can be done so:
    switch_matrix.gnd_input_port(13)
except:
    print('Before executing this make sure that there is no source connected to input 13',
          'as otherwise it will be grounded.\n\n',
          'Then set I_have_made_sure_that_there_is_no_source_connected_to_input_13 = True')
```

#### Set Ground mode ON
The last step to use Ground Mode is to switch it on

```{code-cell} ipython3
switch_matrix.gnd_mode(True)
```

#### Caveats
The port that is selected as *Ground Input Port* cannot be used in normal `connect`/`disconnect` commands

Note: Using the port anyway **will fail** and simply not make the connection. The (dis)connect commands and most (but not all) automatically try to query the status byte and will warn the user if an error occured.

Example:

```{code-cell} ipython3
switch_matrix.reset()
switch_matrix.gnd_mode(True)
```

```{code-cell} ipython3
switch_matrix.connect(12, 22) # 12 is the gnd input port
# A warning should be printed below notifying the user that an error occured
```

```{code-cell} ipython3
# The detailed error message can be retrieved using the `get_error` method:

print(switch_matrix.get_error()) # +3022,"Cannot directory specify Auto Ground Port channel"

# Fun fact: `directory` is a typo in the instrument firmware. It should be "Cannot *directly* ..."
```

#### Ground Mode complete example

```{code-cell} ipython3
switch_matrix.reset()
# After reset:
# - All connections are disconnected,
# - Ground input port is port AUX 12.
# - No inputs or outputs are ground enabled
# - Ground mode is off
# The last two points mean that all inputs and outputs are floating!
```

```{code-cell} ipython3
switch_matrix.unused_inputs([5, 6, 7, 8])
switch_matrix.gnd_enable_output(12)
switch_matrix.gnd_enable_output(33)
# Now
#  - Ground enabled inputs are [5,6,7,8]
#  - Ground enabled Outputs are [12, 33]
# 
# They are not connected to ground yet because Ground Mode is still OFF!
```

```{code-cell} ipython3
switch_matrix.gnd_mode(True)

# Now Inputs [5,6,7,8], and Outputs [12, 13] are connected to ground
# (because they are not connected to anything else)
# 
# All other inputs and outputs are still floating

print(switch_matrix.connections()) # {(12, 12), (12, 33)} # (Only lists connections of output ports!)
```

```{code-cell} ipython3
switch_matrix.connect(4, 33) # Now Output 33 is connected to Input 4 => Output 33 *not* connected to GND

print(switch_matrix.connections()) # {(12, 12), (4, 33)}
```

```{code-cell} ipython3
switch_matrix.disconnect(4,33)
# Output 33 is automatically connected to GND again when the connection is opened again!

print(switch_matrix.connections()) # {(12, 12), (12, 33)}
```

```{code-cell} ipython3
# Ground enabled output ports can also be *Ground disabled* again
# This will disconnect them from GND if they were connected to GND
switch_matrix.gnd_disable_output(12)

print(switch_matrix.connections()) # {(12, 33)}
```

---
### Bias Mode
The Bias Mode is very similar to the previously described Ground Mode, so make sure to read the Ground Mode description above.

The main difference to Ground Mode is that
  - instead of connecting ports to GND potential, in Bias Mode unused Outputs are connected to a Bias Potential which is supplied to the designated *Bias Input Port*.
  - The **default *Bias input Port* is AUX 10**
  - **By default all output ports are *Bias enabled***

Other differences compared to GND mode are:

  - Bias Mode **always requires an external source** to supply the Bias potential (there is no internal circuitry)
  - Only *Output* Ports can be *Bias Enabled* (In Ground mode also Inputs could be *Ground enabled*)

**Note: Bias mode cannot be set to ON when the ground mode is ON.**

**Note: You cannot use the Bias port in `connect`/`disconnect` commands to make explicit connections to the Bias port.**

The other steps are similar to Ground mode so I will jump directly to the complete example:

+++

#### Bias Mode complete Example

```{code-cell} ipython3
switch_matrix.reset()
# After reset:
# - All connections are disconnected,
# - Bias input port is port AUX 10.
# - No inputs or outputs are Bias enabled
# - Bias mode is off
# The last two points mean that all inputs and outputs are floating!
print(switch_matrix.connections()) # {None}
```

```{code-cell} ipython3
# By default all ports are bias enabled!
# For demonstration we'll bias-disable two ports:
switch_matrix.bias_disable_output(15)
switch_matrix.bias_disable_output(17)
#  - Bias disabled Outputs are [15, 17]
# They are not connected to bias input yet because bias Mode is still OFF!
```

```{code-cell} ipython3
print(switch_matrix.connections())
```

```{code-cell} ipython3
switch_matrix.bias_mode(True)
# Now all outputs besides Outputs [15, 17] are connected to bias input port (Port 10 by default)
# 
# Only Outputs [15, 17] will be left floating
print(switch_matrix.connections()) # {(10, 37), (10, 11), (10, 42), (10, 48), (10, 6), (10, 21), (10, 46), (10, 26), (10, 3), (10, 34), (10, 8), (10, 30), (10, 43), (10, 7), (10, 38), (10, 12), (10, 18), (10, 47), (10, 27), (10, 22), (10, 35), (10, 9), (10, 31), (10, 40), (10, 4), (10, 39), (10, 13), (10, 19), (10, 44), (10, 24), (10, 1), (10, 23), (10, 32), (10, 28), (10, 41), (10, 5), (10, 36), (10, 10), (10, 16), (10, 45), (10, 25), (10, 14), (10, 20), (10, 33), (10, 29), (10, 2)}

print((10, 15) in switch_matrix.connections()) # bias-disabled output 15 was not connected to bias port 10
print((10, 17) in switch_matrix.connections()) # bias-disabled output 17 was not connected to bias port 10
```

```{code-cell} ipython3
switch_matrix.connect(4, 11) # Now Output 11 is connected to Input 4 => Output 11 *not* connected to Bias Input

print((10, 11) in switch_matrix.connections()) # False => bias port connection (10, 11) was implicitly opened
print((4, 11) in switch_matrix.connections()) # True => connection (4, 11) was made
```

```{code-cell} ipython3
switch_matrix.disconnect(4, 11)
# Output 11 is automatically connected to Bias Input Port again when the connection is opened again!
# Input 4 is floating

print((10, 11) in switch_matrix.connections()) # True => bias port connection (10, 11) was implicitly closed
print((4, 11) in switch_matrix.connections()) # False => connection (4, 11) was opened
```

```{code-cell} ipython3
# JUst as with GND mode, in bias mode you cannot use the bias input port in the 
# `connect`and `disconnect` commandsto make an explicit connection manually to the bias input:

switch_matrix.connect(10, 17) # Try to make an connection to input port 10 with bias-disabled port 17

print( switch_matrix.get_error() ) # +3014,"Cannot directly specify Bias Port channel"

print((10, 17) in switch_matrix.connections()) # False


# This statement is also valid for bias-enabled ports
switch_matrix.connect(10, 23)  # Try to make an connection to input port 10 with bias-enabled port 23

print( switch_matrix.get_error() ) # +3014,"Cannot directly specify Bias Port channel"

print((10, 17) in switch_matrix.connections()) # False
```

---
### Couple Mode

In *Couple Mode*, pairs of input ports can be treated as one port. This is especially handy for Kelvin Measurement setups (=> *Force* and *Sense* lines), where you connect/disconnect *Force* and *Sense* at the same time from one output to the next.

Couple mode will group subsequent channel numbers in pairs of two:

`{(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (12,13)}`

When making the connections always keep these rule in mind:
  - **Odd** *Input Ports* always connect to an **odd** *Output port*.
    - e.g. Input `3` will connect to `1`, `3`, or `17` and so on
    - Input `3` will never connect to `2`, `6`, `24` and so on
  - **Even** *Input Ports* always connect to an **even** *Output port*.
    - e.g. Input `6` will connect to `2`, `6`, or `34`
    - Input `6` will never connect to `1`, `9`, `21`
  - In a pair the **odd** port is always the lower port number.
    - When `(7,8)` are coupled and sending the command `switch_matrix.connect(7, 24)` the switch matrix 
      - will make the following connection: `7->23`, `8->24`
      - It will **NOT** make `7->24`, `8->25` or `7->24`, `8->23` or 

Because of the above rules it us usually easiest if you always work with the **lower, odd number** in the commands. So for pair `(3,4)` use `3` in the commands.

**Note: There are some limitations on using couple mode together with either Bias or GND mode. Refer to the manual for further information.**

**Note: Couple Ports can also be automagically detected (see examples below)**

**Note: Everytime `couple_ports` is called, the previous setting is overwritten!**

+++

### Enabling couple mode
There are two steps to enabling couple mode:
 - *couple enabled* input ports have to be selected
 - *couple mode* has to be enabled

You can also first enable couple mode and then select couple enabled ports

```{code-cell} ipython3
switch_matrix.reset()
switch_matrix.couple_mode(True) # Enable couple mode

switch_matrix.couple_ports([1]) # Enables couple mode for input pair (1,2)

switch_matrix.couple_ports([1,3,5]) # Enables couple mode for input pairs {(1,2), (3,4), (5,6)}
```

Everytime couple_ports is called, the previous setting is overwritten!

```{code-cell} ipython3
switch_matrix.couple_ports([5])

print("before: ", switch_matrix.couple_ports())

switch_matrix.couple_ports([1, 3])

print("after: ", switch_matrix.couple_ports())
```

#### Making connections in couple mode
Connections in couple mode are made just like in *non-couple* mode, by specifyin **one** input port and **one** output port to connect.

The following examples assume couple mode is enabled for inputs `1`, `3` and `5`

```{code-cell} ipython3
print("before: ", switch_matrix.connections()) # before: {None}

switch_matrix.connect(3, 17)
print("after connecting: ", switch_matrix.connections()) # after connecting:  {(4, 18), (3, 17)}

switch_matrix.disconnect(3, 17)
print("after disconnecting: ", switch_matrix.connections()) # after disconnecting:  {None}
```

```{code-cell} ipython3
# Watch out for the rules mentioned in the introduction to Couple mode above!

switch_matrix.connect(2, 11) # Connections: {(1, 11), (2, 12)} and *NOT* {(2, 11), (1, 12)} or {(2, 11), (1, 10)}
print("after connecting: ", switch_matrix.connections())

switch_matrix.connect(1, 12) # Connections: {(1, 11), (2, 12)} and *NOT* {(1, 12), (2, 11)} or {(1, 12), (2, 13)}
print("after connecting: ", switch_matrix.connections())
```

#### Autodetecting Couple Ports
A very nice feature is that couple ports can also  be autodetected. This assumes that an instrument with Kelvin connections is connected to the input ports.

Setup: connect
 - `SMU +F` to input 3
 - `SMU +S` to input 4
 - `SMU -F` to input 7 
 - `SMU -S` to input 8 

```{code-cell} ipython3
switch_matrix.couple_port_autodetect() 
print("after autodetecting couple ports: ", switch_matrix.connections())
# Will recognize SMU Kelvin connection on Port pairs (3,4) and  (7,8) and enable couple mode for these ports.
```

When using Autodetection, watch out to make the connections correctly according to the *legal* pair numbers `(3,4)` is allowed, `(4,5)` not.

Just for illustration a **bad example**: Connect the SMU as follows:
 - `SMU +F` to input 4  -  **Switch matrix will not detect this**
 - `SMU +S` to input 5  -  **Switch matrix will not detect this**
 - `SMU -F` to input 1  -  **Switch matrix will not detect this**
 - `SMU -S` to input 8  -  **Switch matrix will not detect this**

```{code-cell} ipython3
switch_matrix.couple_port_autodetect() 
print("after autodetecting couple ports: ", switch_matrix.connections())

# Since the Kelvin pairs are not connected to pairs of couple ports, the Switch
# matrix will not *couple enable* these inputs.
```
