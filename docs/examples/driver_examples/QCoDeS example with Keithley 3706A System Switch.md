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

# QCoDeS example with Keithley 3706A System Switch

+++

This is the example notebook that presents the features of the QCoDeS driver for Keithley 3706A-SNFP System Switch with 3730 6x16 High Density Matrix. The model 3706A-SNFP has no DMM capabilities. However, users who are in possession of models with DMM capabilities can use the present driver for switch operations.

+++

## Basic Operation

+++

Let us, first, import QCoDeS driver for the Keithley System Switch:

```{code-cell} ipython3
import qcodes as qc
from qcodes.instrument_drivers.tektronix.Keithley_3706A import Keithley_3706A
```

Now, we create a station to hold our instrument:

```{code-cell} ipython3
station = qc.Station()
```

We finalize the initialization of the instrument by instantiation and then adding it to the station:

```{code-cell} ipython3
smatrix = Keithley_3706A('smatrix', address="USB0::0x05E6::0x3706::04447336::INSTR")
station.add_component(smatrix)
```

Here, the driver not only shows information about the hardware specifications we have, but also reminds us about the interlocks. The channel relays can continue to operate even if the interlock in the corresponding slot is disengaged, however, one cannot perform measurements through the switching card, as the analog backplanes cannot be energized. Specific warnings shall be issued in related function calls as well. Now, let us continue to examine the properties of our instrument via its snapshot:

```{code-cell} ipython3
smatrix.print_readable_snapshot()
```

### Connection types

+++

System switch can be connected to a setup through USB, LAN as well as GPIB. Depending on the needs, one can enable (disable) LAN and/or GPIB connections.

```{code-cell} ipython3
smatrix.gpib_enabled(False)
```

```{code-cell} ipython3
smatrix.lan_enabled(False)
```

```{code-cell} ipython3
smatrix.print_readable_snapshot()
```

```{code-cell} ipython3
smatrix.gpib_enabled(True)
```

```{code-cell} ipython3
smatrix.lan_enabled(True)
```

GPIB connection address can be changed at will. An integer value in between `1` and `30` can be assigned via:

```{code-cell} ipython3
smatrix.gpib_address(12)
```

```{code-cell} ipython3
smatrix.print_readable_snapshot()
```

If an invalid address is passed as an argument, QCoDeS shall rise an error and the present address shall be preserved.

```{code-cell} ipython3
smatrix.gpib_address(31)
```

If the preferred connection method is LAN, one can querry the IP address of the instrument through the driver.

```{code-cell} ipython3
smatrix.get_ip_address()
```

Here, the returned address is the default value if there is no LAN connection; indeed in this example we are connected to the instrument via USB. In cases where a reset to the LAN is required, a simple function call is sufficient:

```{code-cell} ipython3
smatrix.reset_local_network()
```

### Identifications

+++

While QCoDeS prints out the instrument identifications including the installed matrix types upon a successfull connection, these metadata can be queried later on, as well.

```{code-cell} ipython3
smatrix.get_idn()
```

```{code-cell} ipython3
smatrix.get_switch_cards()
```

### Save and load setups, memory management

+++

The system switch provides an option for the users to save their current setups to be used later. One can save a configuration either to the non-volatile memory of the instrument or to a USB drive. To this end, we use `save_setup()` function which either takes no arguments or a string which represents the path and the file name to which the setup shall be saved on an external drive. If no arguments provided, the setup shall be saved to the intstument. Note that, in the latter case, any existing saved configurations shall be overwritten. 

```{code-cell} ipython3
smatrix.save_setup()
```

We can recall saved setups via `load_setup()` function. It takes a single argument: If it is `0`, factory defaults load, if it is `1`, the saved setup from the nonvolatile memory is recalled. Otherwise, a string specifying the relative path to the saved setup on a USB drive should be passed in.

```{code-cell} ipython3
smatrix.load_setup(0)
```

```{code-cell} ipython3
smatrix.load_setup(1)
```

The system switch has limited memory which can be queried via:

```{code-cell} ipython3
smatrix.get_available_memory()
```

The memory usage of a saved setup is depicted by the `Config Memory`. User given channel patterns occupy the `Pattern Memory`. The system switch is capable of storing and running Lua scripts. This functionality is not explicitly implemented by the QCoDeS driver. However, if desired, one can still send, save and execute Lua scripts through `write()` and `ask()` methods of the `InstrumentBase` class of QCoDeS which is inherited by every instrument driver. If this is the case, the saved Lua scripts occupy the `Script Memory`.    

+++

## Channel Control and Manipulations

+++

### Channel Specifiers

+++

First, let us call the installed switch cards again:

```{code-cell} ipython3
smatrix.get_switch_cards()
```

The system switch has six available slots from which three are currently occupied by `6x16` high density matrices. As implied, these matrices have `6` rows and `16` columns, hence, in total, `96` available channels, each. 

Each channel has identical properties but unique names. The properties of the channels can be manipulated via calling them individually, as a group or as a whole slot. The naming of the channels follows a general pattern. In particular, here for the model `3730` which has no `bank`, the names starts with the slot number, then the row number and finally follows the cloumn number which should always be two characters. For example, `1101` represents the channel at `Slot 1`, `Row 1` and `Column 1`. For other models, users should refer to the corresponding instrument manual.

In certain cases manual deduction of the name of the channel could be more efficient. On the other hand, user can easily query the available channels as follows:

```{code-cell} ipython3
smatrix.get_channels()
```

The `get_channels()` function returns an array of strings each being the name of an available channel in all occupied slots. The return type being `string` is intentional as the instrument accepts the channel names as strings and not integers during any manipulation. Thus, any element(s) of this list can be safely passed as an argument to another function which specifies a channel property. 

If desired, the available channels in a given slot can be queried, as well.

```{code-cell} ipython3
smatrix.get_channels_by_slot(1)
```

### Backplane Specifiers

+++

Each matrix has typically six analog backplane relays which can be associated which a channel(s). The naming scheme of the relays are similar to that of the channels. Each name start with the slot number, continues with `9`, which is the unique backplane number, and then finishes with backplane relay identifier which can take the values `11`, `12`, ..., `16`. That is, we can select the backplane relay `12` on `Slot 3` via `3912`. 

Unless a backplane is associated with a channel, the instrument does not return the names of the relays. This is overcomed within the QCoDeS driver, so that user can list the available analog backplane relays.

```{code-cell} ipython3
smatrix.get_analog_backplane_specifiers()
```

### Control and manipulations

+++

The switch channels as well as the analog backplane relays can be controlled via a set of well defined functions. 
First, a channel can be opened and closed, as expected.

```{code-cell} ipython3
smatrix.open_channel("1101")
```

We can provide a list of channels as an argument. In this case, channel names should be provided as a single string, seperated with a comma and no blank spaces.

```{code-cell} ipython3
smatrix.open_channel("1103,2111")
```

We can provide a channel range via slicing. The following call opens the channels from `3101` to `3116`.

```{code-cell} ipython3
smatrix.open_channel("3101:3116")
```

We can open all the channels in a slot by passing `slotX` where `X=1,...,6` representing the slot id. In our case, `X` can be `1`, `2` or `3`.

```{code-cell} ipython3
smatrix.open_channel("slot2")
```

Finally, we can open all channels in all slots via:

```{code-cell} ipython3
smatrix.open_channel("allslots")
```

Let us reset everything to factory defaults.

```{code-cell} ipython3
smatrix.reset_channel("allslots")
```

The `reset_channel()` function can take the name of a single channel, a channel list, range or a slot, as well.

We continue with closing the desired channels. In this case, one cannot pass slot ids or `allslots` as the argument; it is not possible to close slots all together, simultaneously.

```{code-cell} ipython3
smatrix.close_channel("1101")
```

```{code-cell} ipython3
smatrix.close_channel("2111,3216")
```

We can query the closed channels.

```{code-cell} ipython3
smatrix.get_closed_channels("allslots")
```

```{code-cell} ipython3
smatrix.get_closed_channels("slot2")
```

We can exclusively close channels, as well. In one of the two ways to achieve exclusive close of channels, we specify channels to be closed in such a way that any presently closed channels in all slots open if they are not included to list. 

Currently we know that the channels `1101`, `2111` and `3216` are closed. Now, let us close the channel `3101` exclusively and then query close channels. We expect to only see the latter and the former channels should be opened.

```{code-cell} ipython3
smatrix.exclusive_close("3101")
```

```{code-cell} ipython3
smatrix.get_closed_channels("allslots")
```

Second way to exclusively close channels is similar to that of our previous example. We now exclusively close the specified channels on the associated slots. Other channels are opened if they are not specified by the parameter within the same slot, while the other slots remain untouched. Currently the only closed channel is `3101`. Let close, exclusively, `2101` and `2216`. Also, let us close the channel `2111` genericaly. We expect the channel `2111` opens while the channel `3101` remains closed.

```{code-cell} ipython3
smatrix.close_channel("2111")
```

```{code-cell} ipython3
smatrix.exclusive_slot_close("2101,2216")
```

```{code-cell} ipython3
smatrix.get_closed_channels("allslots")
```

We can control connection rules for closing and opening channels when using the `exclusive_close()` and `exlusive_slot_close()` functions. The three valid rules are `BREAK_BEFORE_MAKE`, `MAKE_BEFORE_BREAK` and `OFF`. The default rule is `BREAK_BEFORE_MAKE`.

```{code-cell} ipython3
smatrix.channel_connect_rule()
```

If the connection rule is set to `BREAK_BEFORE_MAKE`, it is ensured that all channels open before any channels close. If it is set to `MAKE_BEFORE_BREAK`, it is ensured that all channels close before any channels open. Finally, if it is set to `OFF`, channels open and close, simultaneously.

```{code-cell} ipython3
smatrix.channel_connect_rule("MAKE_BEFORE_BREAK")
```

```{code-cell} ipython3
smatrix.channel_connect_rule()
```

```{code-cell} ipython3
smatrix.channel_connect_rule("OFF")
```

```{code-cell} ipython3
smatrix.channel_connect_rule()
```

Note that, resetting channels to factory defaults will not change the connection rule:

```{code-cell} ipython3
smatrix.reset_channel("allslots")
```

```{code-cell} ipython3
smatrix.channel_connect_rule()
```

```{code-cell} ipython3
smatrix.channel_connect_rule("BREAK_BEFORE_MAKE")
```

```{code-cell} ipython3
smatrix.channel_connect_rule()
```

In certain cases, we may want to keep certain channels always open. We can achieve desired behavior by setting the specified channels (and analog backplane relays) as `forbidden` to close.

```{code-cell} ipython3
smatrix.set_forbidden_channels("1101:1105")
```

```{code-cell} ipython3
smatrix.get_forbidden_channels("allslots")
```

These channels cannot be closed:

```{code-cell} ipython3
smatrix.close_channel("1101")
```

```{code-cell} ipython3
smatrix.get_closed_channels("slot1")
```

We can forbid entire slots to be closed. In this case, the analog backplane relays of the corresponding slot shall be flagged as forbidden, as well.

```{code-cell} ipython3
smatrix.set_forbidden_channels("slot2")
```

```{code-cell} ipython3
smatrix.get_forbidden_channels("allslots")
```

We can clear the forbidden list when we desire to do so.

```{code-cell} ipython3
smatrix.clear_forbidden_channels("slot1")
```

```{code-cell} ipython3
smatrix.get_forbidden_channels("allslots")
```

```{code-cell} ipython3
smatrix.clear_forbidden_channels("2911,2912,2916")
```

```{code-cell} ipython3
smatrix.get_forbidden_channels("slot2")
```

```{code-cell} ipython3
smatrix.clear_forbidden_channels("slot2")
```

```{code-cell} ipython3
smatrix.get_forbidden_channels("allslots")
```

Finally, we can set additional delay times to specified channels. The delays should be provided in seconds.

```{code-cell} ipython3
smatrix.set_delay("1101", 1)
```

```{code-cell} ipython3
smatrix.get_delay("1101")
```

```{code-cell} ipython3
smatrix.set_delay("slot2", 2)
```

```{code-cell} ipython3
smatrix.get_delay('slot2')
```

We can use `reset_channel()` function to set delay times to defalult value of `0`

```{code-cell} ipython3
smatrix.reset_channel("allslots")
```

```{code-cell} ipython3
smatrix.get_delay("allslots")
```

We can, conveniently, connect (disconnect) given rows (columns) of a slot to a column (row) of the same slot via the functions `connect_or_disconnect_row_to_columns()` and `connect_or_disconnect_column_to_rows()`. Each function takes an action as its first argument. The action should be either `connect` or `disconnect`. The slot id is provided as the second argument. The `connect_or_disconnect_row_to_columns()` takes the row number as the third argument to which the desired cloumns, specified by the final argument as a list, will be connected. Likewise, the `connect_or_disconnect_column_to_rows()` takes the column number as the third argument to which the desired rows, specified by the final argument as a list, will be connected.   

Both functions opens (closes) the formed channels automatically and returns their list which can be used for later use, if desired. Let us, first, connect `1`, `2` and `13` columns of `slot2` to the row `3` of the same slot.  

```{code-cell} ipython3
channels = smatrix.connect_or_disconnect_row_to_columns("connect", 2, 3, [1, 2, 13])
```

```{code-cell} ipython3
channels
```

The channels of the matrix returned are now opened. Let us flag these channels as forbidden to close:

```{code-cell} ipython3
for channel in channels:
    smatrix.set_forbidden_channels(channel)
```

```{code-cell} ipython3
smatrix.get_forbidden_channels("allslots")
```

Similarly, let us disconnect the rows `1`, `2` and `4` of `slot1` to the column `16` of the same slot:

```{code-cell} ipython3
channels = smatrix.connect_or_disconnect_column_to_rows("disconnect", 1, 16, [1, 2, 4])
```

```{code-cell} ipython3
channels
```

```{code-cell} ipython3
smatrix.get_closed_channels("slot1")
```

### Association of Backplanes

+++

We can set specific analog backplane relays to desired channels so that they can be utilized in switching applications. Note that the driver will correctly warn us as the slots are disengaged.

```{code-cell} ipython3
smatrix.set_backplane("1101:1109", "1916")
```

```{code-cell} ipython3
smatrix.get_backplane("1101")
```

```{code-cell} ipython3
smatrix.get_backplane("slot1")
```

We can clear the association via resetting the corresponding channels (or all channels).

```{code-cell} ipython3
smatrix.reset_channel("1101:1109")
```

```{code-cell} ipython3
smatrix.get_backplane("slot1")
```

## Exceptions

+++

The instrument itself does not return any error messages if it encounters an exception, but, rather silently handles such situations. Thus, QCoDeS driver is equipped with validation and exception handling with informative error messages for probable scenerios. 

Here, we note some important cases where errors are generated if:

    -  A non existing channel or slot passed as an argument to a function
    -  Slots are passed as arguments of `close_channel()`, `exclusive_close()` and `exclusive_slot_close()`
    -  A delay time is tried to be set for analog backplane relays
    -  A channel name is used to set as a backplane instead of a backplane name in `set_backplane()` and vice versa

```{code-cell} ipython3

```
