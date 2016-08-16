import qcodes as qc
from qcodes import Instrument
from qcodes.utils.validators import Numbers, Ints, Enum, Strings
from time import sleep
from imp import reload

if __name__ == "__main__":
    import qcodes.instrument_drivers.AlazarTech.ATS9440 as ATS_driver
    import qcodes.instrument_drivers.AlazarTech.ATS_acquisition_controllers as ATS_control

    try:
        ATS.close()
        ATS_controller.close()
    except:
        pass
    reload(ATS_driver)
    reload(ATS_control)

    ATS = ATS_driver.ATS9440('ATS', server_name='Alazar_server')
    ATS_controller = ATS_control.Average_AcquisitionController(name='ATS_control',
                                                               alazar_id=0,
                                                               server_name='Alazar_server')

    ATS.config(trigger_source1='CHANNEL_C',
               trigger_level1=135,
               channel_range=2,
               sample_rate=1e6,
               coupling='DC')
    ATS_controller.set_acquisitionkwargs(buffer_timeout=5000,
                                         samples_per_record=12000,
                                         records_per_buffer=4,
                                         buffers_per_acquisition=1,
                                         channel_selection='AC')
    result = ATS_controller.do_acquisition()