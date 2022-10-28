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

# QCoDeS Example with Tektronix Keithley S46

The S46 is an RF swicth with four relays "A" to "D". These relays either have four or six poles, depending on the instrument model. Each pole constitutes a channel and can either be "open" or "closed". Channel "A1" is attached to the first pole on relay "A", channel "B2" is attached to the second pole of relay "B", etc...

Channels "A1" to "D6" are all "normally open". Only one channel per relay may be closed.

Additionally, there are optionally eight relays "R1" to "R8" which are two pole relays. One pole is "normally closed" and the other "normally open". For these relays, we have one channel per relay, so "R1" is both a channel and a relay. Upon closing the channel, the normally open pole will close.

In this notebook, we have verified with a multi-meter that channels indeed open and close as expected.

Note: We have performed tests with a six pole instrument. Although it is expected that this driver should work with a four pole instrument, this has not been verified due to a lack of instrument availability

```{code-cell} ipython3
from qcodes.instrument_drivers.tektronix.Keithley_s46 import S46, LockAcquisitionError
```

```{code-cell} ipython3
s46 = S46("s2", "GPIB0::7::INSTR")
```

```{code-cell} ipython3
print(s46.available_channels)
print(len(s46.available_channels))
```

```{code-cell} ipython3
s46.closed_channels()
```

```{code-cell} ipython3
s46.open_all_channels()
```

```{code-cell} ipython3
s46.closed_channels()
```

```{code-cell} ipython3
s46.A1()
```

```{code-cell} ipython3
s46.A1("close")
```

```{code-cell} ipython3
s46.A1()
```

```{code-cell} ipython3
s46.closed_channels()
```

```{code-cell} ipython3
try:
    s46.A2("close")
    raise("We should not be here")
except LockAcquisitionError as e:
    print(e)
```

```{code-cell} ipython3
s46.A1("open")
```

```{code-cell} ipython3
s46.A2("close")
```

```{code-cell} ipython3
try:
    s46.A1("close")
    raise("We should not be here")
except LockAcquisitionError as e:
    print(e)
```

```{code-cell} ipython3
s46.B1("close")
```

```{code-cell} ipython3
try:
    s46.B2("close")
    raise("We should not be here")
except LockAcquisitionError as e:
    print(e)
```

```{code-cell} ipython3
s46.closed_channels()
```

```{code-cell} ipython3
s46.open_all_channels()
```

```{code-cell} ipython3
s46.closed_channels()
```
