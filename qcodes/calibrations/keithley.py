import time

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter

src_FS_map = {"200e-3": 180e-3, "2": 1.8, "20": 18, "200": 180}


def setup_dmm(dmm: Instrument) -> None:
    dmm.aperture_time(1.0)
    dmm.autozero("OFF")
    dmm.autorange("OFF")


def save_calibration(smu: Instrument) -> None:
    for smu_channel in smu.channels:
        smu.write(f"{smu_channel.channel}.cal.save()")


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
    for smu_channel in smu.channels:
        input(f"Please connect channel {smu_channel.channel} to V input on calibrated DMM.")
        for smu_range, dmm_range in zip(smu_ranges, dmm_ranges):
            dmm.range(dmm_range)
            calibrate_keithley_smu_v_single(
                smu, smu_channel.channel, dmm.volt, smu_range, src_Z, time_delay
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
    assert channel in set(smu_channel.channel for smu_channel in smu.channels)
    assert v_range in src_FS_map.keys()
    src_FS = src_FS_map[v_range]

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

        smu.write(f"{channel}.cal.polarity = {channel}.CAL_AUTO")

        # Reset the smu to default state
        smu.write(f"{channel}.source.levelv = {src_Z}")
