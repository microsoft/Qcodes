#%% Function definitions
import time
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter


src_FS_map = {"200e-3": 180e-3, "2": 1.8, "20": 18, "200": 180}
channels = ["smua", "smub"]


def setup_dmm(dmm: Instrument) -> None:
    dmm.aperture_time(1.0)
    dmm.autozero("OFF")
    dmm.autorange("OFF")


def save_calibration(smu: Instrument) -> None:
    for channel in channels:
        smu.write(f"{channel}.cal.save()")


def calibrate_keithley_smu_v(
    smu: Instrument,
    dmm: Instrument,
    src_Z: float = 1e-30,
    time_delay: float = 3.0,
    save_calibrations: bool = False,
) -> None:
    smu_ranges = ["200e-3", "2", "20"]
    dmm_ranges = [1, 10, 100]
    setup_dmm(dmm)
    for channel in channels:
        input(f"Please connect channel {channel} to V input on calibrated DMM.")
        for smu_range, dmm_range in zip(smu_ranges, dmm_ranges):
            dmm.range(dmm_range)
            calibrate_keithley_smu_v_single(
                smu, channel, dmm.volt, smu_range, src_Z, time_delay
            )
    if save_calibrations:
        save_calibration(smu)


def calibrate_keithley_smu_v_single(
    smu: Instrument,
    channel: str,
    dmm_param_volt: Parameter,
    v_range: str,
    src_Z: float = 1e-30,
    time_delay: float = 3.0,
) -> None:
    assert channel in channels
    assert v_range in src_FS_map.keys()
    src_FS = src_FS_map[v_range]

    if channel == "smua":
        smun = smu.smua
    elif channel == "smub":
        smun = smu.smub

    sense_modes = ["SENSE_LOCAL"]

    for sense_mode in sense_modes:
        print("Sense mode: " + sense_mode)
        smu.write(f'{channel}.cal.unlock("KI0026XX")')
        smu.write(f"{channel}.reset()")
        smu.write(f"{channel}.source.func = {channel}.OUTPUT_DCVOLTS")

        smu.write(f"{channel}.source.rangev = {v_range}")
        smu.write(f"{channel}.source.output = {channel}.OUTPUT_OFF")
        smu.write(f"{channel}.sense = {channel}." + sense_mode)
        time.sleep(time_delay)

        # Start positive calibration:
        smu.write(f"{channel}.cal.polarity = {channel}.CAL_POSITIVE")
        smu.write(f"{channel}.source.levelv = {src_Z}")
        smu.write(f"{channel}.source.output = {channel}.OUTPUT_ON")

        # Measure positive zero voltage with SMU and DMM:
        time.sleep(time_delay)
        smu.write(f"Z_rdg = {channel}.measure.v()")
        DMM_Z_rdg = dmm_param_volt()

        smu.write(f"{channel}.source.output = {channel}.OUTPUT_OFF")
        smu.write(f"{channel}.source.levelv = {src_FS:.8e}")
        smu.write(f"{channel}.source.output = {channel}.OUTPUT_ON")

        # Measure positive full scale voltage with SMU and DMM:
        time.sleep(time_delay)
        smu.write(f"FS_rdg = {channel}.measure.v()")
        DMM_FS_rdg = dmm_param_volt()

        # Write positive v_range calibration to SMU:
        smu.write(f"{channel}.source.output = {channel}.OUTPUT_OFF")
        time.sleep(time_delay)
        smu.write(
            f"{channel}.source.calibratev({v_range}, {src_Z}, {DMM_Z_rdg:.8e}, {src_FS:.8e}, {DMM_FS_rdg:.8e})"
        )
        if sense_mode != "SENSE_CALA":
            time.sleep(time_delay)
            smu.write(
                f"{channel}.measure.calibratev({v_range}, Z_rdg, {DMM_Z_rdg:.8e}, FS_rdg, {DMM_FS_rdg:.8e})"
            )
        time.sleep(time_delay)

        # Debug output
        print(
            f"{channel}.source.calibratev({v_range}, {src_Z}, {DMM_Z_rdg:.8e}, {src_FS:.8e}, {DMM_FS_rdg:.8e})"
        )
        print(
            f"{channel}.measure.calibratev({v_range}, Z_rdg, {DMM_Z_rdg:.8e}, FS_rdg, {DMM_FS_rdg:.8e})"
        )

        # Start negative calibration:
        smu.write(f"{channel}.cal.polarity = {channel}.CAL_NEGATIVE")
        smu.write(f"{channel}.source.levelv = -{src_Z}")
        smu.write(f"{channel}.source.output = {channel}.OUTPUT_ON")

        # Measure negative zero voltage with SMU and DMM:
        time.sleep(time_delay)
        smu.write(f"Z_rdg = {channel}.measure.v()")
        DMM_Z_rdg = dmm_param_volt()

        smu.write(f"{channel}.source.output = {channel}.OUTPUT_OFF")
        smu.write(f"{channel}.source.levelv = -{src_FS:.8e}")
        smu.write(f"{channel}.source.output = {channel}.OUTPUT_ON")

        # Measure negative full scale voltage with DMM:
        time.sleep(time_delay)
        smu.write(f"FS_rdg = {channel}.measure.v()")
        DMM_FS_rdg = dmm_param_volt()

        # Write negative v_range calibration to SMU:
        smu.write(f"{channel}.source.output = {channel}.OUTPUT_OFF")
        time.sleep(time_delay)
        smu.write(
            f"{channel}.source.calibratev(-{v_range}, -{src_Z}, {DMM_Z_rdg:.8e}, -{src_FS:.8e}, {DMM_FS_rdg:.8e})"
        )
        if sense_mode != "SENSE_CALA":
            time.sleep(time_delay)
            smu.write(
                f"{channel}.measure.calibratev(-{v_range}, Z_rdg, {DMM_Z_rdg:.8e}, FS_rdg, {DMM_FS_rdg:.8e})"
            )
        time.sleep(time_delay)

        # Debug output
        print(
            f"{channel}.source.calibratev(-{v_range}, -{src_Z}, {DMM_Z_rdg:.8e}, -{src_FS:.8e}, {DMM_FS_rdg:.8e})"
        )
        print(
            f"{channel}.measure.calibratev(-{v_range}, Z_rdg, {DMM_Z_rdg:.8e}, FS_rdg, {DMM_FS_rdg:.8e})"
        )

        time.sleep(time_delay)

        smu.write(f"{channel}.cal.polarity = {channel}.CAL_AUTO")

        # Reset the smu to default state
        smu.write(f"{channel}.source.levelv = {src_Z}")


#%% Load instruments (the SMU is assumed to be imported)
from qcodes.instrument_drivers.Keysight.Keysight_34470A_submodules import (
    Keysight_34470A,
)

dmm = Keysight_34470A("dmm", "TCPIP0::10.164.54.211::inst0::INSTR")

#%% Define current experiment
load_or_create_experiment(
    "MeasurementSetupDebug", context.devices[-1].name, load_last_duplicate=True
)

# %% calibrate both channels

calibrate_keithley_smu_v(smu, dmm)
smu.smua.volt(0)
smu.smub.volt(0)

#%% calibrate single channel in specific range

setup_dmm(dmm)
dmm.range(1.0)
calibrate_keithley_smu_v_single(smu, "smua", dmm.volt, "200e-3")

# smu.smua.volt(0)

# %% check calibration by measuring voltage with dmm while sweeping with keithley
smun = smu.smua

smun.sourcerange_v(200e-3)
smun.measurerange_v(200e-3)
smun.volt(0)
smun.output("on")

dmm.aperture_time(0.1)
dmm.range(1)
dmm.autozero("OFF")
dmm.autorange("OFF")

do1d(
    smun.volt,
    -0.1e-3,
    0.1e-3,
    101,
    0.3,
    dmm.volt,
    smun.curr,
    measurement_name="smu debug",
)

smun.volt(0)
smun.output("off")
#%% save calibrations
save_calibration(smu)
