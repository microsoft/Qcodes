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
    import qcodes.instrument_drivers.AlazarTech.ATS9440 as ATS_driver
    import qcodes.instrument_drivers.AlazarTech.ATS_acquisition_controllers as ATS_controller_driver

    ATS = ATS_driver.ATS9440('ATS', server_name='Alazar_server')
    ATS_controller = ATS_controller_driver.Average_AcquisitionController(name='ATS_control',
                                                                         alazar_name='ATS',
                                                                         server_name='Alazar_server')

    # Configure ATS and ATS_controller
    ATS.config(trigger_source1='CHANNEL_C',
               trigger_level1=135,
               channel_range=2,
               sample_rate=1e6,
               coupling='DC')

    read_length = 0.03
    ATS_controller.average_mode('none')
    samples_per_record = int(16 * round(float(ATS.sample_rate() * read_length) / 16))
    ATS_controller.update_acquisition_kwargs(buffer_timeout=5000,
                                             samples_per_record=samples_per_record,
                                             records_per_buffer=1,
                                             buffers_per_acquisition=100,
                                             channel_selection='AC')

    from meta_instruments import Analysis
    lre_analysis = Analysis.LoadReadEmptyAnalysis('LRE_analysis', ATS_controller=ATS_controller)
    lre_analysis.load_duration(5)
    lre_analysis.read_duration(20)
    lre_analysis.empty_duration(5)
    print(lre_analysis.fidelity())