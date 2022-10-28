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

# QCoDeS Example with Tektronix AWG70002A

The Tektronix awg70002A can operate in two modes: function generator mode or AWG mode. This example notebook briefly covers both.

```{code-cell} ipython3
%matplotlib notebook
import matplotlib.pyplot as plt

import numpy as np
from qcodes.instrument_drivers.tektronix.AWG70002A import AWG70002A
```

```{code-cell} ipython3
awg = AWG70002A('awg', 'TCPIP0::172.20.2.243::inst0::INSTR')
```

```{code-cell} ipython3
# Let's have a look at the available parameters

awg.print_readable_snapshot(update=True)
```

## Function Generator Style Operation

```{code-cell} ipython3
# Set the intrument mode to function generator
awg.mode('FGEN')

# Build some signal
awg.ch1.fgen_type('EXPONENTIALDECAY')
awg.ch1.fgen_frequency(10e6)
awg.ch1.fgen_amplitude(0.074)
awg.ch1.fgen_offset(0.12)
awg.ch1.fgen_phase(25)

# Switch channel 1 on
awg.ch1.state(1)

# Start outputting...
awg.play()
```

```{code-cell} ipython3
# switch off the output eventually
awg.stop()

# and disable the channel
awg.ch1.state(0)
```

## AWG Style Operation

The instrument can be operated as an awg where the user uploads arrays describing the waveforms.

Each channel operates in one of three resolution modes:

* 8 bit signal + 2 markers
* 9 bit signal + 1 marker
* 10 bit signal with no markers

Waveforms can be sent to the waveform list via `.wfmx` files. A `.wfmx` file can contain marker data. If the resolution of the instrument does not allow for markers, these are simply ignored.

+++

### Making and sending waveforms to the waveform list

```{code-cell} ipython3
# set the instrument in awg mode
awg.mode('AWG')
# set the resolution to 8 bits plus two markers
awg.ch1.resolution(8)
```

```{code-cell} ipython3
# clear the sequence list and waveform list (NOT ALWAYS A GOOD IDEA! BE CAREFUL!)
awg.clearSequenceList()
awg.clearWaveformList()
```

```{code-cell} ipython3
# Let us make a sine, upload it and play it

N = 50000  # minimal length allowed is 2400 points

m1 = np.concatenate((np.ones(int(N/2)), np.zeros(int(N/2))))
m2 = np.concatenate((np.zeros(int(N/2)), np.ones(int(N/2))))

ramp = 0.075*np.linspace(0, 1, N)

mysine = 0.1*np.sin(10*2*np.pi*np.linspace(0, 1, N)) + ramp

data = np.array([mysine, m1, m2])
```

```{code-cell} ipython3
# The .wfmx file needs a name in the memory of the instrument
# The name of the waveform in the waveform list is that same name
# with no .wfmx extension
filename = 'examplewaveform1.wfmx'
```

```{code-cell} ipython3
# now compile the binary file
wfmx_file = awg.makeWFMXFile(data, 0.350)
```

```{code-cell} ipython3
# and send it and load it into memory
awg.sendWFMXFile(wfmx_file, filename)
awg.loadWFMXFile(filename)
```

```{code-cell} ipython3
# The waveform is now in the waveform list
awg.waveformList
```

```{code-cell} ipython3
# now assign it to channel 1
awg.ch1.setWaveform(filename.replace('.wfmx', ''))
```

```{code-cell} ipython3
# Switch channel 1 on
awg.ch1.state(1)

# Start outputting...
awg.play()
```

```{code-cell} ipython3
# switch off the output eventually
awg.stop()

# and disable the channel
awg.ch1.state(0)
```

```{code-cell} ipython3
awg.ch2.setWaveform(filename.replace('.wfmx', ''))
```

```{code-cell} ipython3
awg.ch2.state(0)
```

## Making and sending sequences

Sequences are much better off being generated using the broadbean module, but for now let's reduce the number of moving parts and compose a little sequence by hand.

```{code-cell} ipython3
# set the instrument in awg mode
awg.mode('AWG')
# set the resolution to 8 bits plus two markers
awg.ch1.resolution(8)
```

```{code-cell} ipython3

# Let's make a sequence where a sine plays on one channel while the other channel ramps
# and then the roles reverse

# As a preparation, let's set both channels to 300 mV peak-to-peak
awg.ch1.awg_amplitude(0.3)
awg.ch2.awg_amplitude(0.3)

N = 20000  # minimally 2400

SR = 1e9
awg.sample_rate(SR)  # set the sample rate on the instrument
ramp_target = 0.1  # ramp target (V)

time = np.linspace(0, N/SR, N)
sinesignal = 0.15*np.sin(SR/N*2*np.pi*time)
m1 = np.concatenate((np.ones(int(N/2)), np.zeros(int(N/2))))
m2 = np.concatenate((np.zeros(int(N/2)), np.zeros(int(N/2))))
rampsignal = np.linspace(0, ramp_target, N)

# Then we compose and upload a .seqx file in 6 steps

# Step 1: cast the waveform data into the .wfmx format
# To make a .wfmx, we need to know the amplitude of the output channel
ch1_amp = awg.ch1.awg_amplitude()
ch2_amp = awg.ch2.awg_amplitude()

#wfm_ch1_n1 = awg.makeWFMXFile(np.array([sinesignal, m1, m2]), ch1_amp)
#wfm_ch1_n2 = awg.makeWFMXFile(np.array([rampsignal, m1, m2]), ch1_amp)
#wfm_ch2_n1 = awg.makeWFMXFile(np.array([rampsignal, m1, m2]), ch2_amp)
#wfm_ch2_n2 = awg.makeWFMXFile(np.array([sinesignal, m1, m2]), ch2_amp)


wfm_ch1_n1 = np.array([sinesignal, m1, m2])
wfm_ch1_n2 = np.array([rampsignal, m1, m2])
wfm_ch2_n1 = np.array([rampsignal, m1, m2])
wfm_ch2_n2 = np.array([sinesignal, m1, m2])

# Step 2: decide on sequencing information
# This information is provided as lists of the same length as the
# sequence
trig_waits = [0, 0]  # 0: off, 1: trigA, 2: trigB, 3: EXT
nreps = [2, 3]  # 0 corresponds to infinite
event_jumps = [0, 0] # 0: off, 1: trigA, 2: trigB, 3: EXT
event_jump_to = [0, 0]  # irrelevant if event-jump is 0, else the sequence pos. to jump to
go_to = [0, 1]  # 0 means next

# Step 3: make the .seqx file
# The sequence must be given a name

seqname = 'tutorial_sequence'

wfms = [[wfm_ch1_n1, wfm_ch1_n2], [wfm_ch2_n1, wfm_ch2_n2]]

seqx = awg.makeSEQXFile(trig_waits,
                        nreps,
                        event_jumps,
                        event_jump_to,
                        go_to,
                        wfms,
                        [ch1_amp, ch2_amp],
                        seqname)

# Step 4: Transfer the seqx file
awg.sendSEQXFile(seqx, 'thursday.seqx')

# Step 5: Load the seqx file
awg.loadSEQXFile('thursday.seqx')
# Now the sequence should appear in the sequencelist, but it is not yet assigned to channels

# Step 6: Assign tracks from the sequence to the channels
# Unlike older/other AWG models, this can be done on a per-channel basis
awg.ch1.setSequenceTrack(seqname, 1)
awg.ch2.setSequenceTrack(seqname, 2)
```

```{code-cell} ipython3
# Now play it!
awg.ch1.state(1)
awg.ch2.state(1)
awg.play()
```

```{code-cell} ipython3
awg.stop()
```

```{code-cell} ipython3
# Finally irreversibly tear down the instrument
awg.close()
```
