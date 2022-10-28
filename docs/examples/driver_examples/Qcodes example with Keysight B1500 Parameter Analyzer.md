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

+++ {"toc": true}

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Qcodes-example-with-Keysight-B1500-Semiconductor-Parameter-Analyzer" data-toc-modified-id="Qcodes-example-with-Keysight-B1500-Semiconductor-Parameter-Analyzer-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Qcodes example with Keysight B1500 Semiconductor Parameter Analyzer</a></span><ul class="toc-item"><li><span><a href="#Instrument-Short-info" data-toc-modified-id="Instrument-Short-info-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Instrument Short info</a></span><ul class="toc-item"><li><span><a href="#Physical-grouping" data-toc-modified-id="Physical-grouping-1.1.1"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>Physical grouping</a></span></li><li><span><a href="#Logical-grouping" data-toc-modified-id="Logical-grouping-1.1.2"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>Logical grouping</a></span></li></ul></li><li><span><a href="#Qcodes-driver-info" data-toc-modified-id="Qcodes-driver-info-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Qcodes driver info</a></span><ul class="toc-item"><li><span><a href="#Integer-Flags-and-Constants-used-in-the-driver" data-toc-modified-id="Integer-Flags-and-Constants-used-in-the-driver-1.2.1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span>Integer Flags and Constants used in the driver</a></span></li><li><span><a href="#High-level-interface" data-toc-modified-id="High-level-interface-1.2.2"><span class="toc-item-num">1.2.2&nbsp;&nbsp;</span>High level interface</a></span></li><li><span><a href="#Low-level-interface" data-toc-modified-id="Low-level-interface-1.2.3"><span class="toc-item-num">1.2.3&nbsp;&nbsp;</span>Low level interface</a></span></li></ul></li><li><span><a href="#Programming-Examples" data-toc-modified-id="Programming-Examples-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Programming Examples</a></span><ul class="toc-item"><li><span><a href="#Initializing-the-instrument" data-toc-modified-id="Initializing-the-instrument-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>Initializing the instrument</a></span></li></ul></li><li><span><a href="#High-Level-Interface" data-toc-modified-id="High-Level-Interface-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>High Level Interface</a></span><ul class="toc-item"><li><span><a href="#Identifying-and-selecting-installed-modules" data-toc-modified-id="Identifying-and-selecting-installed-modules-1.4.1"><span class="toc-item-num">1.4.1&nbsp;&nbsp;</span>Identifying and selecting installed modules</a></span></li><li><span><a href="#Enabling-/-Disabling-channels" data-toc-modified-id="Enabling-/-Disabling-channels-1.4.2"><span class="toc-item-num">1.4.2&nbsp;&nbsp;</span>Enabling / Disabling channels</a></span></li><li><span><a href="#Perform-self-calibration" data-toc-modified-id="Perform-self-calibration-1.4.3"><span class="toc-item-num">1.4.3&nbsp;&nbsp;</span>Perform self calibration</a></span></li><li><span><a href="#Performing-sampling-measurements" data-toc-modified-id="Performing-sampling-measurements-1.4.4"><span class="toc-item-num">1.4.4&nbsp;&nbsp;</span>Performing sampling measurements</a></span></li><li><span><a href="#CV-Sweep" data-toc-modified-id="CV-Sweep-1.4.5"><span class="toc-item-num">1.4.5&nbsp;&nbsp;</span>CV Sweep</a></span></li><li><span><a href="#IV-Sweep" data-toc-modified-id="IV-Sweep-1.4.6"><span class="toc-item-num">1.4.6&nbsp;&nbsp;</span>IV Sweep</a></span></li><li><span><a href="#Performing-phase-compensation" data-toc-modified-id="Performing-phase-compensation-1.4.7"><span class="toc-item-num">1.4.7&nbsp;&nbsp;</span>Performing phase compensation</a></span></li><li><span><a href="#Performing-Open/Short/Load-correction" data-toc-modified-id="Performing-Open/Short/Load-correction-1.4.8"><span class="toc-item-num">1.4.8&nbsp;&nbsp;</span>Performing Open/Short/Load correction</a></span><ul class="toc-item"><li><span><a href="#Set-and-get-reference-values" data-toc-modified-id="Set-and-get-reference-values-1.4.8.1"><span class="toc-item-num">1.4.8.1&nbsp;&nbsp;</span>Set and get reference values</a></span></li><li><span><a href="#Add-CMU-output-frequency-to-the-list-for-correction" data-toc-modified-id="Add-CMU-output-frequency-to-the-list-for-correction-1.4.8.2"><span class="toc-item-num">1.4.8.2&nbsp;&nbsp;</span>Add CMU output frequency to the list for correction</a></span></li><li><span><a href="#Clear-CMU-output-frequency-list" data-toc-modified-id="Clear-CMU-output-frequency-list-1.4.8.3"><span class="toc-item-num">1.4.8.3&nbsp;&nbsp;</span>Clear CMU output frequency list</a></span></li><li><span><a href="#Query-CMU-output-frequency-list" data-toc-modified-id="Query-CMU-output-frequency-list-1.4.8.4"><span class="toc-item-num">1.4.8.4&nbsp;&nbsp;</span>Query CMU output frequency list</a></span></li><li><span><a href="#Open/Short/Load-Correction" data-toc-modified-id="Open/Short/Load-Correction-1.4.8.5"><span class="toc-item-num">1.4.8.5&nbsp;&nbsp;</span>Open/Short/Load Correction</a></span></li></ul></li><li><span><a href="#SMU-sourcing-and-measuring" data-toc-modified-id="SMU-sourcing-and-measuring-1.4.9"><span class="toc-item-num">1.4.9&nbsp;&nbsp;</span>SMU sourcing and measuring</a></span></li><li><span><a href="#Setting-up-ADCs-to-NPLC-mode" data-toc-modified-id="Setting-up-ADCs-to-NPLC-mode-1.4.10"><span class="toc-item-num">1.4.10&nbsp;&nbsp;</span>Setting up ADCs to NPLC mode</a></span></li></ul></li><li><span><a href="#Error-Message" data-toc-modified-id="Error-Message-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Error Message</a></span></li><li><span><a href="#Low-Level-Interface" data-toc-modified-id="Low-Level-Interface-1.6"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>Low Level Interface</a></span></li></ul></li></ul></div>

+++

# Qcodes example with Keysight B1500 Semiconductor Parameter Analyzer

+++

## Instrument Short info
Here a short introduction on how the B1500 measurement system is composed is given. For a detailed overview it is strongly recommended to refer to the *B1500 Programming Guide* and also the *Parametric Measurement Handbook* by Keysight.

### Physical grouping
The Keysight B1500 Semiconductor Parameter Analyzer consists of a *Mainframe* and can be equipped with various instrument *Modules*. 10 *Slots* are available in which up to 10 *modules* can be installed (some *modules* occupy two *slots*). Each *module* can have one or two *channels*.

### Logical grouping
The measurements are typically done in one of the 20 measurement modes. The modes can be roughly subdivided into
  - Spot measurements
      - **High Speed Spot Measurements**
  - Pulsed Spot measurement
  - Sweep Measurements
  - Search Measurements

The **High Speed Spot (HSS)** Mode is essentually just a fancy way of saying to take readings and forcing constant voltages/currents. The *HSS* commands work at any time, independent of the currenttly selected Measurment Mode.

With the exception of the *High Speed Spot Measurement Mode*, the other modes have to be activated and configured by the user.

+++

## Qcodes driver info
As can be seen already from the instrument short info, the instrument is very versatile, but also very complex. Hence the driver will eventually consist of two layers:
  - The Low Level interface allows one to utilize all functions of the driver by offering a thin wrapper around the FLEX command set that the B1500 understands.
  - A Higher Level interface that provides a convenient access to the more frequently used features. Not all features are available via the high level interface.

The two driver levels can be used at the same time, so even if some functionality is not yet implemented in the high-level interface, the user can send a corresponding low-level command.

### Integer Flags and Constants used in the driver
Both the high-level and the low-level interface use integer constants in many commands. For user convienience, the `qcodes.instrument_drivers.Keysight.keysightb1500.constants` provides more descriptive Python Enums for these constants. Although bare integer values can still be used, it is highly recommended to use the enumerations in order to avoid mistakes.

### High level interface
The high level exposes instrument functionality via QCodes Parameters and Python methods on the mainframe object and the individual instrument module objects. For example, *High Speed Spot* Measurement commands for forcing constant voltages/currents or for taking simple readings are implemented.

### Low level interface
The Low Level interface (`MessageBuilder` class) provides a wrapper function for each FLEX command. From the low-level, the full functionality of the instrument can be controlled.

The `MessageBuilder` assembles a message string which later can be sent to the instrument using the low level `write` and `ask` methods. One can also use the `MessageBuilder` to write FLEX complex measurement routines that are stored in the B1500 and can be executed at a later point. This can be done to enable fast execution.

+++

## Programming Examples

+++

### Initializing the instrument

```{code-cell} ipython3
from matplotlib import pyplot as plt
import qcodes as qc

from qcodes.instrument_drivers.Keysight.keysightb1500 import KeysightB1500, \
    MessageBuilder, constants
from qcodes.dataset.experiment_container import load_or_create_experiment
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.plotting import plot_dataset
from qcodes import initialise_database
from pyvisa.errors import VisaIOError

from IPython.display import display, Markdown
```

```{code-cell} ipython3
station = qc.Station() # Create a station to hold all the instruments
```

```{code-cell} ipython3
#Note: If there is no physical instrument connected
#the following code will try to load a simulated instrument

try:
    #TODO change that address according to your setup
    b1500 = KeysightB1500('spa', address='GPIB21::17::INSTR')
    display(Markdown("**Note: using physical instrument.**"))
except (ValueError, VisaIOError):
    # Either there is no VISA lib installed or there was no real instrument found at the
    # specified address => use simulated instrument
    import qcodes.instrument.sims as sims
    path_to_yaml = sims.__file__.replace('__init__.py',
                                         'keysight_b1500.yaml')

    b1500 = KeysightB1500('SPA',
                          address='GPIB::1::INSTR',
                          visalib=path_to_yaml + '@sim')
    display(Markdown("**Note: using simulated instrument. Functionality will be limited.**"))
```

```{code-cell} ipython3
station.add_component(b1500)
```

## High Level Interface

Here is an example of using high-level interface.

+++

### Identifying and selecting installed modules
As mentioned above, the B1500 is a modular instrument, and contains multiple cards. When initializing the driver, the driver requests the installed modules from the B1500 and exposes them to the user via multiple ways.

The first way to address a certain module is e.g. as follows:

```{code-cell} ipython3
b1500.smu1 # first SMU in the system
b1500.cmu1 # first CMU in the system
b1500.smu2 # second SMU in the system
```

```{code-cell} ipython3
b1500.cmu1.phase_compensation_mode()
```

The naming scheme is - `b1500.<instrument class as lower case><number>`, where number is `1` for the first instrument in its class, `2` for the second instrument in its class and so on. (*Not the channel or slot number!*)

Next to this direct access - which is simple and good for direct user interaction - the modules are also exposed via multiple data structures through which they can be adressed:
  - by slot number
  - by module kind (such as SMU, or CMU)
  - by channel number

This can be more convenient for programmatic selection of the modules.

+++

Instrument modules are installed in slots (numbered 1-11) and can be selected by the slot number:

```{code-cell} ipython3
b1500.by_slot
```

All modules are also grouped by module kind (see `constants.ModuleKind` for list of known kinds of modules):

```{code-cell} ipython3
b1500.by_kind
```

For example, let's list all SMU modules:

```{code-cell} ipython3
b1500.by_kind['SMU']
```

Lastly, there is dictionary of all module channels:

```{code-cell} ipython3
# For the simulation driver:
# Note how the B1530A module has two channels.
# The first channel number is the same as the slot number (6).
# The second channel has a `02` appended to the channel number.
b1500.by_channel
```

**Note: For instruments with only one channel, channel number is the same as the slot number. However there are instruments with 2 channels per card. For these instruments the second channel number will differ from the slot number.**

**Note for the simulated instrument: The simulation driver will list a B1530A module with 2 channels as example.**

In general, the slot- and channel numbers can be passed as integers. However (especially in the case of the channel numbers for multi-channel instruments) it is recommended to use the Python enums defined in `qcodes.instrument_drivers.Keysight.keysightb1500.constants`:

```{code-cell} ipython3
# Selecting a module by channel number using the Enum
m1 = b1500.by_channel[constants.ChNr.SLOT_01_CH1]

# Without enum
m2 = b1500.by_channel[1]

# And we assert that we selected the same module:
assert m1 is m2
```

### Enabling / Disabling channels

Before sourcing or doing a measurement, the respective channel has to be enabled. There are two ways to enable/disable a channel:
  - By directly addressing the module
  - By addressing the mainframe and specifying which channel(s) to be enabled

The second method is useful if multiple channels shall be enabled, or for programmatic en-/disabling of channels. It also allows to en-/disable all channels with one call.

```{code-cell} ipython3
# Direct addressing the module
b1500.smu1.enable_outputs()
b1500.smu1.disable_outputs()
```

```{code-cell} ipython3
# Enabling via the mainframe

# enable one channel
b1500.enable_channels([1])

# enable multiple channels
b1500.enable_channels([1, 2])

# disable multiple channels
b1500.disable_channels([1, 2])

# disable all channels
b1500.disable_channels()
```

### Perform self calibration

Calibration takes about 30 seconds (the visa timeout for it is controlled by `b1500.calibration_time_out` attribute).

```{code-cell} ipython3
b1500.self_calibration()
```

### Performing sampling measurements

+++

This section outlines steps to perform sampling measurement.

+++

Set a sample rate and number of samples.

```{code-cell} ipython3
# Number of spot measurments made per second and stored in a buffer.
sample_rate = 0.02
# Total number of spot measurements.
nsamples = 100
```

Assign timing parameters to SMU.

```{code-cell} ipython3
b1500.smu1.timing_parameters(0, sample_rate, nsamples)
```

Autozero is generally disabled for sampling measurement.

```{code-cell} ipython3
b1500.autozero_enabled(False)
```

Set SMU to sampling mode.

```{code-cell} ipython3
b1500.smu1.measurement_mode(constants.MM.Mode.SAMPLING)
```

SMU is configured with by assigning voltage output range, input output range and compliance. While forcing voltage, current should be the compliance and vice versa.

```{code-cell} ipython3
b1500.smu1.source_config(output_range=constants.VOutputRange.AUTO,
                       compliance=1e-7,
                       compl_polarity=None,
                       min_compliance_range=constants.IOutputRange.AUTO
                      )
```

Set the averaging to 1 otherwise the measurement takes 10 times more time.

```{code-cell} ipython3
b1500.use_nplc_for_high_speed_adc(n=1)
```

Set the voltage

```{code-cell} ipython3
b1500.smu1.enable_outputs()
b1500.smu1.voltage(1e-6)
```

We are now ready to start the sampling measurement. We first initialize the database and create-new/load-old experiment. Then we register our dependent and independent parameters and start the measurement.

**Note** that the default values of label and units are not defined for the parameter sampling measurement trace. Hence we first set them according to what is being measured: in this case we will measure current in A. It is important to set the label and the unit before the measurement in order to have this information when looking at the acquired data, for example when plotting it with `plot_dataset` as shown below.

```{code-cell} ipython3
b1500.smu1.sampling_measurement_trace.label = 'Current'
b1500.smu1.sampling_measurement_trace.unit = 'A'
# Automatic assignment of the label and unit based on
# the settings of the instrument can be implemented
# upon request.
```

```{code-cell} ipython3
initialise_database()
exp = load_or_create_experiment(experiment_name='dummy_sampling_measurement',
                          sample_name="no sample")
meas  = Measurement(exp=exp)
meas.register_parameter(b1500.smu1.sampling_measurement_trace)

with meas.run() as datasaver:
    datasaver.add_result((b1500.smu1.sampling_measurement_trace, b1500.smu1.sampling_measurement_trace.get()))
```

Thanks to the `label` and `unit` set above for the `sampling_measurement_trace` parameter, the `plot_dataset` function is able to produce a plot with a useful label for the vertical axis, see below:

```{code-cell} ipython3
plot_dataset(datasaver.dataset)
```

Check compliance: For the values which are compliant the output is one and for others it is zero. A quick to visualize of your measurements are compliant is to plot the compliance data and look if any value of zero.

```{code-cell} ipython3
data_compliance = b1500.smu1.sampling_measurement_trace.compliance()
```

```{code-cell} ipython3
plt.plot(data_compliance)
plt.xlabel('Measurements')
_ = plt.ylabel('Compliance status')
```

The channel number of the measured data can be obtained in the following way.

```{code-cell} ipython3
data_channel = b1500.smu1.sampling_measurement_trace.data.channel
data_channel[:5]
```

If you want to know the type of the measured data, for ex 'I' or 'V' the following method can be used.

```{code-cell} ipython3
data_type = b1500.smu1.sampling_measurement_trace.data.type
data_type[:5]
```

The measurement status can be obtained using:

```{code-cell} ipython3
data_status = b1500.smu1.sampling_measurement_trace.data.status
data_status[:5]
```

The variable 'data_status' is a list of strings of measurement status for each data point. One can look at the meaning of the statuses in `constants.MeasurementStatus` class. It enlists meaning of all possible measurement status. For example: in case the measurement status is 'C' its meaning can be found as following.

```{code-cell} ipython3
constants.MeasurementStatus.N
```

```{code-cell} ipython3
constants.MeasurementStatus.C
```

### CV Sweep

+++

MFCMU has two modes of measurement. The first is spot measurement and this here is sweep measurement. As the name suggest sweep measurement execute the measurement once for the whole list of voltages and saves the output in the buffer untill measurment is completed.

The function below sets up properly the parameters to run the sweep measurements. Look at the docstring of ``setup_staircase_cv`` to know more about each argument of the function.

```{code-cell} ipython3
b1500.cmu1.enable_outputs()
```

```{code-cell} ipython3
b1500.cmu1.setup_staircase_cv(
    v_start=0,
    v_end=1,
    n_steps=201,
    freq=1e3,
    ac_rms=250e-3,
    post_sweep_voltage_condition=constants.WMDCV.Post.STOP,
    adc_mode=constants.ACT.Mode.PLC,
    adc_coef=5,
    imp_model=constants.IMP.MeasurementMode.Cp_D,
    ranging_mode=constants.RangingMode.AUTO,
    fixed_range_val=None,
    hold_delay=0,
    delay=0,
    step_delay=225e-3,
    trigger_delay=0,
    measure_delay=0,
    abort_enabled=constants.Abort.ENABLED,
    sweep_mode=constants.SweepMode.LINEAR,
    volt_monitor=False)
```

If the setup function does not output any error then we are ready for the measurement.

```{code-cell} ipython3
initialise_database()
exp = load_or_create_experiment(
    experiment_name='dummy_capacitance_measurement',
    sample_name="no sample"
)
meas = Measurement(exp=exp)

meas.register_parameter(b1500.cmu1.run_sweep)

with meas.run() as datasaver:
    res = b1500.cmu1.run_sweep()
    datasaver.add_result((b1500.cmu1.run_sweep,res))
```

The ouput of the ``run_sweep`` is a primary parameter (Capacitance) and a secondary parameter (Dissipation). The type of primary and secondary parameter  depends on the impedance model set in the ``setup_staircase_cv`` function (or via the corresponding ``impedance_model`` parameter). The setpoints of both the parameters are the same voltage values as defined by ``setup_staircase_cv`` (behind the scenes, those values are available in the ``cv_sweep_voltages`` parameter).

```{code-cell} ipython3
plot_dataset(datasaver.dataset)
```

```{code-cell} ipython3
b1500.cmu1.run_sweep.status_summary()
```

### IV Sweep

+++

This section explains the IV Staircase sweep measurements.

+++

Enable the channels.

```{code-cell} ipython3
b1500.smu1.enable_outputs()
b1500.smu2.enable_outputs()
```

```{code-cell} ipython3
# Always good to do for the safety of the measured sample
b1500.smu2.voltage(0)
b1500.smu1.voltage(0)
```

Setting up smu1 and smu2 for running the staircase sweep. One of the smu's is used for sweep. Both the smu's are used for acquiring data. It is possible to acquire data with more SMUs, that depends on the measurement mode (see below), so refer to the instrument manual for information on how many 'channels' can be measured with which measurement mode. In the setup below, smu1 is used to sweep over the sweep voltages -3 to 3 in 201 steps.

```{code-cell} ipython3
b1500.smu1.setup_staircase_sweep(
    v_src_range = constants.VOutputRange.AUTO,
    v_start = 3,
    v_end = -3,
    n_steps = 201,
    av_coef = 5,
    step_delay = 0.225,
    abort_enabled = constants.Abort.ENABLED,
    i_meas_range = constants.IMeasRange.FIX_10nA,
    i_comp = 1e-8,
    sweep_mode = constants.SweepMode.LINEAR,
    # and there are more arguments with default values
    # that might need to be changed for your
    # particular measurement situation
)
```

`smu2` is kept at constant voltage and at different compliance and measurement range settings.

```{code-cell} ipython3
b1500.smu2.voltage(10e-3)
b1500.smu2.enable_filter(True)
b1500.smu2.measurement_operation_mode(constants.CMM.Mode.COMPLIANCE_SIDE)
b1500.smu2.current_measurement_range(constants.IMeasRange.FIX_10uA)
```

`b1500.set_measurement_mode` is used to define measurement mode and the channels from which data is extracted from. Here, channels correspond to SMU1 and SMU2 respectively - the SMU which is setup to run the sweep needs to go **FIRST**.

```{code-cell} ipython3
b1500.set_measurement_mode(
    mode=constants.MM.Mode.STAIRCASE_SWEEP,
    channels=(b1500.smu1.channels[0], b1500.smu2.channels[0])
)
# SMUs have only one channel so using `channels[0]` is enough
# This might be improved in the future for better clarity and user convenience.
```

`run_iv_staircase_sweep` is used to run the sweep

```{code-cell} ipython3
initialise_database()
exp = load_or_create_experiment(
    experiment_name='dummy_iv_sweep_measurement',
    sample_name="no sample"
)
meas = Measurement(exp=exp)

# As per user needs, names and labels of the parameters inside the
# MultiParameter can be adjusted to reflect what is actually being
# measured using the convenient `set_names_labels_and_units` method.
# The setpoint name/label/unit (the independent sweep 'parameter')
# can be additionally customized using `set_setpoint_name_label_and_unit`
# method.
# Below is an example of using `set_names_labels_and_units`:
b1500.run_iv_staircase_sweep.set_names_labels_and_units(
    names=('gate_current', 'source_drain_current'),
    labels=('Gate current', 'Source-drain current'),
)
# The number of names (and labels) MUST be the same as the number of channels,
# and the order of the names should match the order of channels, as passed to
# `set_measurement_mode` method.

meas.register_parameter(b1500.run_iv_staircase_sweep)

with meas.run() as datasaver:
    res = b1500.run_iv_staircase_sweep()
    datasaver.add_result((b1500.run_iv_staircase_sweep, res))

# In production code, remeber to revert the names/labels of the
# run_iv_staircase_sweep MultiParameter in order to avoid confusion.
```

```{code-cell} ipython3
plot_dataset(datasaver.dataset)
```

```{code-cell} ipython3
b1500.run_iv_staircase_sweep.status_summary()
```

### Performing phase compensation

+++

The phase compensation is performed to adjust the phase zero.

One must take care of two things before executing the phase compensation. First, make sure that all the channel outputs are enabled else instrument throws an error.

```{code-cell} ipython3
b1500.run_iv_staircase_sweep.measurement_status()
```

Second, the phase compensation mode must be set to manual.

```{code-cell} ipython3
b1500.cmu1.phase_compensation_mode(constants.ADJ.Mode.MANUAL)
```

Now the phase compensation can be performed as follows. This operation takes about 30 seconds (the visa timeout for this operation is set via `b1500.cmu1.phase_compensation_timeout` attribute).

```{code-cell} ipython3
b1500.cmu1.phase_compensation()
```

Note that `phase_compensation` method also supports loading data of previously performed phase compensation. To use that, explicitly pass the operation mode argument:

```{code-cell} ipython3
b1500.cmu1.phase_compensation(constants.ADJQuery.Mode.USE_LAST)
```

### Performing Open/Short/Load correction

+++

#### Set and get reference values

+++

Use the following method to set the calibration values or reference values of the open/short/load standard. Here, we are using open correction with Cp-G mode. The primary reference value, which is the value for Cp (in F), is set to 0.00001, and the secondary reference value, which is the value of G (in S), is set to 0.00002. These values are completely arbitrary, so please change them according to your experiments.

```{code-cell} ipython3
b1500.cmu1.correction.set_reference_values(corr=constants.CalibrationType.OPEN,
                                           mode=constants.DCORR.Mode.Cp_G,
                                           primary=0.00001,
                                           secondary=0.00002)
```

You can retrieve the values you have set for calibration or the reference values of the open/short/load standard in the following way:

```{code-cell} ipython3
b1500.cmu1.correction.get_reference_values(corr=constants.CalibrationType.OPEN)
```

#### Add CMU output frequency to the list for correction

+++

You can add to the list of frequencies supported by the instrument to be used for the data correction. The frequency value can be given with a certain resolution as per Table 4-18 in the programming manual.

```{code-cell} ipython3
b1500.cmu1.correction.frequency_list.add(1000)
```

#### Clear CMU output frequency list

+++

Clear the frequency list for the correction data measurement using the following methods. Correction data will be invalid after calls to these methods, so you will have to again perform the open/short/load correction.

There are two modes in which you can clear the frequency list. First is clearing the list of frequencies:

```{code-cell} ipython3
b1500.cmu1.correction.frequency_list.clear()
```

Second is clearing the list of frequencies and also setting it to a default list of frequencies (for the list of default frequencies, refer to the documentation of the ``CLCORR`` command in the programming manual):

```{code-cell} ipython3
b1500.cmu1.correction.frequency_list.clear_and_set_default()
```

#### Query CMU output frequency list

It is possible to query the total number of frequencies in the list:

```{code-cell} ipython3
b1500.cmu1.correction.frequency_list.query()
```

It is also possible to query the values of specific frequencies using the same method by specifying an index within the frequency list:

```{code-cell} ipython3
b1500.cmu1.correction.frequency_list.query(2)
```

#### Open/Short/Load Correction

+++

As per description in the programming guide, we first set the oscillator level of the CMU output signal.

```{code-cell} ipython3
# Set oscillator level
b1500.cmu1.voltage_ac(30e-3)
```

To perform open/short/load correction connect the open/short/load standard and execute the following command to perform and enable the correction.

```{code-cell} ipython3
b1500.cmu1.correction.perform_and_enable(corr=constants.CalibrationType.OPEN)
#b1500.cmu1.correction.perform_and_enable(corr=constants.CalibrationType.SHORT)
#b1500.cmu1.correction.perform_and_enable(corr=constants.CalibrationType.LOAD)
```

In case you would only like to perform the correction but not enable it, you can use separate methods `perform` and `enable`.

+++

To check whether a correction is enabled, use the following method:

```{code-cell} ipython3
b1500.cmu1.correction.is_enabled(corr=constants.CalibrationType.OPEN)
```

To disable a performed correction, use the following method:

```{code-cell} ipython3
b1500.cmu1.correction.disable(corr=constants.CalibrationType.OPEN)
```

### SMU sourcing and measuring
The simplest measurement one can do with the B1500 are *High Speed Spot Measurements*. They work independent of the selected *Measurement Mode*.

The `voltage` and `current` Qcodes Parameters that the SMU High Level driver exposes will execute *High Speed Spot* measurements. Additionally, there are functions that let the user specify the output/measure ranges, and compliance limits.

To source a voltage/current do the following:
  1. Configure source range, and (optionally) compliance settings
  2. Enable the channel
  3. Force the desired voltage
  4. (optionally) Disable the channel

**Note: The source settings (Step 1) are persistent until changed again. So for sucessive measurements the configuration can be omitted.**

```{code-cell} ipython3
b1500.smu1.enable_outputs()

b1500.smu1.source_config(output_range=constants.VOutputRange.AUTO,
                         compliance=0.1)

b1500.smu1.voltage(1.5)
```

To measure do the following:
  1. Configure the voltage or/and current measure ranges
  2. Enable the channel (if not yet enabled)
  3. Do the measurement
  4. (optionally) Disable the channel

**Note: The measure settings (Step 1) are persistent until changed again. So for sucessive measurements the configuration can be omitted.**

```{code-cell} ipython3
b1500.smu1.i_measure_range_config(i_measure_range=constants.IMeasRange.MIN_100mA)
b1500.smu1.v_measure_range_config(v_measure_range=constants.VMeasRange.FIX_2V)

b1500.smu1.enable_outputs()

cur = b1500.smu1.current()
vol = b1500.smu1.voltage()

b1500.smu1.disable_outputs()
```

### Setting up ADCs to NPLC mode

Both the mainframe driver and SMU driver implement convenience methods for controlling integration time of the *High Speed Spot* measurement, which allow setting ADC type, and setting the frequenty used NPLC mode.

+++

Use the following methods on the mainframe instance to set up the ADCs to NPLC mode:

```{code-cell} ipython3
# Set the high-speed ADC to NPLC mode,
# and optionally specify the number of PLCs as an arugment
# (refer to the docstring and the user manual for more information)
b1500.use_nplc_for_high_speed_adc(n=1)

# Set the high-resolution ADC to NPLC mode,
# and optionally specify the number of PLCs as an arugment
# (refer to the docstring and the user manual for more information)
b1500.use_nplc_for_high_resolution_adc(n=5)
```

And then use the following methods on the SMU instances to use particular ADC for the particular SMU:

```{code-cell} ipython3
# Use high-speed ADC
# with the settings defined above
# for the SMU 1
b1500.smu1.use_high_speed_adc()

# Use high-resoultion ADC
# with the settings defined above
# for the SMU 2
b1500.smu2.use_high_resolution_adc()
```

## Error Message

The error messages from the instrument can be read using the following method. This method reads one error code from the head of the error queue and removes that code from the queue. The read error is returned as the response of this method.

```{code-cell} ipython3
b1500.error_message()
```

Here, the response message contains an error number and an error message. In some cases the error message may also contain the additional information such as the slot number. They are separated by a semicolon (;). For example, if the error 305 occurs on the slot 1, this method returns the following response. 305,"Excess current in HPSMU.; SLOT1"

If no error occurred, this command returns 0,"No Error".

+++

## Low Level Interface

The Low Level Interface provides a wrapper around the FLEX command set. Multiple commands can be assembled in a sequence. Finally, the command sequence is compiled into a command string, which then can be sent to the instrument.

Only some very minimal checks are done to the command string. For example some commands have to be the *last* command in a sequence of commands because the fill the output queue. Adding additional commands after that is not allowed.

As an example, a "voltage source + current measurement" is done, similar as was done above with the high level interface.

```{code-cell} ipython3
mb = MessageBuilder()

mb.cn(channels=[1])
mb.dv(chnum=1,
      voltage=1.5,
      v_range=constants.VOutputRange.AUTO,
      i_comp=0.1)
mb.ti(chnum=1,
      i_range=constants.IMeasRange.FIX_100uA)
mb.cl(channels=[1])

# Compiles the sequence of FLEX commands into a message string.
message_string = mb.message
```

```{code-cell} ipython3
print(message_string)
```

The message string can be sent to the instrument. To parse the response of this spot measurement command, use the `KeysightB1500.parse_spot_measurement_response` static method.

`parse_spot_measurement_response` will return a `dict` that contains the measurement value together with the measurement channel, info on what was measured (current, voltage, capacitance, ...), and status information. For a detailed description, see the user manual.

```{code-cell} ipython3
response = b1500.ask(message_string)

KeysightB1500.parse_spot_measurement_response(response)
```

The `MessageBuilder` object can be cleared, which allows the object to be reused to generate a new message string.

```{code-cell} ipython3
mb.clear_message_queue()
# This will produce empty string because MessageBuilder buffer was cleared
mb.message
```

The `MessageBuilder` provides a *fluent* interface, which means every call on the `MessageBuilder` object always returns the object itself, with the exeption of `MessageBuilder.message` which **returns the compiled message string**.

This means that the same message as in the first example could've been assembled like this:

```{code-cell} ipython3
response = b1500.ask(
    MessageBuilder()
    .cn(channels=[1])
    .dv(
        chnum=1,
        voltage=1.5,
        v_range=constants.VOutputRange.AUTO,
        i_comp=0.1,
    )
    .ti(chnum=1, i_range=constants.IMeasRange.FIX_100uA)
    .cl(channels=[1])
    .message
)

KeysightB1500.parse_spot_measurement_response(response)
```
