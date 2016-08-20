import qcodes as qc
import numpy as np
from qcodes import Instrument
from qcodes.utils.validators import Numbers, Ints, Enum, Strings
from time import sleep
from imp import reload
import qcodes.instrument.parameter as parameter
from qcodes import Instrument

loc_provider = qc.data.location.FormatLocation(fmt='data/{date}/#{counter}_{name}_{time}')
qc.data.data_set.DataSet.location_provider = loc_provider

mode = 'ATS'

if __name__ == "__main__":
    if mode == 'test':
        # Finally show that this instrument also works within a loop
        dummy = parameter.ManualParameter(name="dummy")
        dummy_detect = parameter.StandardParameter(name='dummy_detect', names=('dummy_detect', 'dummy_detect2'),
                                                   get_cmd=lambda: [123,124],shapes=((),()))
        data = qc.Loop(dummy[0:3:1]).each(dummy_detect).run(name='DummyTest', use_threads=False, background=False,
                                                            quiet=True)

        print(data.dummy_detect)
    elif mode == 'dummy':
        import qcodes.tests.instrument_mocks as mocks
        ins = mocks.DummyInstrument()
        ins.add_names(['asd', 'dfg'])
        ins.dac1.names
    else:
        import qcodes.instrument_drivers.AlazarTech.ATS9440 as ATS_driver
        import qcodes.instrument_drivers.AlazarTech.ATS_acquisition_controllers as ATS_controller_driver
        ATS = ATS_driver.ATS9440('ATS', server_name='Alazar_server')
        ATS_controller = ATS_controller_driver.Average_AcquisitionController(name='ATS_control',
                                                                             alazar_name='ATS',
                                                                             server_name='Alazar_server')

        ATS.config(trigger_source1='CHANNEL_C',
                   trigger_level1=135,
                   channel_range=2,
                   sample_rate=1e6,
                   coupling='DC')
        ATS_controller.average_mode('point')
        ATS_controller.set_acquisitionkwargs(buffer_timeout=5000,
                                             samples_per_record=50000,
                                             records_per_buffer=1,
                                             buffers_per_acquisition=1,
                                             channel_selection='AC')
        p = ATS_controller.acquisition.names


        dummy_sweep = parameter.ManualParameter(name="dummy")
        # data = qc.Loop(TGAC.sweep(1.62,1.625,0.001), delay=0.003).each(

        data = qc.Loop(dummy_sweep[1:3:1], delay=0.003).each(
            ATS_controller.acquisition).run(name='testsweep', background=False)

        data.ATS_control_acquisition