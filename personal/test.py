import qcodes as qc
from qcodes import Instrument
from qcodes.utils.validators import Numbers, Ints, Enum, Strings
from time import sleep
from imp import reload

if __name__ == "__main__":
    import qcodes.instrument_drivers.AlazarTech.ATS9440 as ATS_driver
    import qcodes.instrument_drivers.AlazarTech.ATS_acquisition_controllers as ATS_control

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
                                         samples_per_record=1600,
                                         records_per_buffer=5,
                                         buffers_per_acquisition=2,
                                         channel_selection='A')
    ATS_controller.average_mode('point')

    import qcodes.instrument_drivers.stanford_research.SIM900 as SIM900_driver

    SIM900 = SIM900_driver.SIM900('SIM900', 'GPIB0::4::INSTR')
    TG, LB, RB, TGAC, SRC, _, DS, DF = [eval('SIM900.chan{}'.format(i)) for i in range(1, 9)]

    loc_provider = qc.data.location.FormatLocation(fmt='data/{date}/#{counter}_{name}_{time}')
    qc.data.data_set.DataSet.location_provider = loc_provider

    data = qc.Loop(TGAC.sweep(1.62, 1.625, 0.001), delay=0.003).each(
        ATS_controller.acquire).run(name='testsweep', background=False)
    print(data.ATS_control_acquire)