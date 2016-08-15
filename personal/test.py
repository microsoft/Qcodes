import qcodes as qc
from qcodes import Instrument
from qcodes.utils.validators import Numbers, Ints, Enum, Strings
from time import sleep

if __name__ == "__main__":
    import qcodes.instrument_drivers.AlazarTech.ATS9440 as ATS_driver
    import qcodes.instrument_drivers.AlazarTech.ATS_acquisition_controllers as ATS_control

    ATS = ATS_driver.ATS9440('ATS', server_name='Alazar_server')
    ATS_controller = ATS_control.Average_AcquisitionController(name='ATS_control',
                                                               alazar_id=0,
                                                               server_name='Alazar_server')

    ATS.config(trigger_source1='EXTERNAL',
               trigger_level1=135,
               trigger_engine2='TRIG_ENGINE_K',
               channel_range=[2, 2],
               sample_rate=10e6)
    ATS_controller.set_acquisitionkwargs(buffer_timeout=1000,
                                         samples_per_record=20000,
                                         records_per_buffer=100,
                                         buffers_per_acquisition=2)

    result = ATS_controller.do_acquisition()