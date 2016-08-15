import qcodes as qc
from qcodes import Instrument
from qcodes.utils.validators import Numbers, Ints, Enum, Strings
from time import sleep

if __name__ == "__main__":
    from qcodes.instrument_drivers.AlazarTech import ATS_acquisition_controllers
    from qcodes.instrument_drivers.AlazarTech import ATS9440


    ATS_acquisition_controller = ATS_acquisition_controllers.Basic_AcquisitionController()