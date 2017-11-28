import time


import qcodes
from qcodes.instrument_drivers.stanford_research.SR860 import SR860
from qcodes.instrument_drivers.QuTech.IVVI import IVVI

def main():
    sr = SR860("sr", "GPIB0::4::INSTR")
    ivvi = IVVI("ivvi", "COM4")

    sr.buffer.capture_config("X,Y")

    def send_trigger():
        ivvi.trigger()
        time.sleep(0.1)

    n_samples = 100
    sr.buffer.start_capture("ONE", "SAMP")

    time.sleep(0.1)
    for _ in range(n_samples):
        send_trigger()

    sr.buffer.stop_capture()
    sr.buffer.get_capture_data(n_samples)
    meas = qcodes.Measure(sr.buffer.X)
    data = meas.run()


if __name__ == "__main__":
    main()
