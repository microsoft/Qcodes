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

# QCoDeS Example with Minicircuits Switch Boxes Controlled via USB

```{code-cell} ipython3
from qcodes.instrument_drivers.Minicircuits.USB_SPDT import USB_SPDT
```

change the serial number to the serial number on the sticker on the back of the device, or leave it blank if there is only one switch box connected

The driver_path should specify the url of the dll for controlling the instrument. You can find it here:

https://ww2.minicircuits.com/softwaredownload/rfswitchcontroller.html

Download .NET dll and save somewhere. Unblock it (right click properties) and specify the path.

```{code-cell} ipython3
dev = USB_SPDT('test',
               serial_number='11703020018',
               driver_path= r'C:\Users\a-dovoge\Qcodes\qcodes\instrument_drivers\Minicircuits\mcl_RF_Switch_Controller64')
```

setting value to line one or two

```{code-cell} ipython3
dev.a(1)
```

reading value

```{code-cell} ipython3
dev.b()
```

setting all switches to line 2

```{code-cell} ipython3
dev.all(2)
```
