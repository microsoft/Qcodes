import qcodes as qc
import numpy as np
from qcodes import Instrument
from qcodes.utils.validators import Numbers, Ints, Enum, Strings
from time import sleep
from imp import reload
import qcodes.instrument.parameter as parameter
from qcodes import Instrument
import os

loc_provider = qc.data.location.FormatLocation(fmt='data/{date}/#{counter}_{name}_{time}')
qc.data.data_set.DataSet.location_provider = loc_provider

mode = 'ATS'

#
# def configure_ArbStudio(ArbStudio, voltages, channel_factors, marker_cycles=100):
#     stages = ['empty', 'load', 'read']
#
#     ArbStudio.ch4_clear_waveforms()
#     for ch in [1, 2, 3]:
#         eval("ArbStudio.ch{}_trigger_source('fp_trigger_in')".format(ch))
#         eval("ArbStudio.ch{}_trigger_mode('stepped')".format(ch))
#         eval('ArbStudio.ch{}_clear_waveforms()'.format(ch))
#         waveforms = channel_factors[ch - 1] * np.array([[voltages[stage]] * 10 for stage in stages])
#         for waveform in waveforms:
#             eval('ArbStudio.ch{}_add_waveform(waveform)'.format(ch))
#
#         eval('ArbStudio.ch{}_sequence([0, 1, 2])'.format(ch))
#
#     waveforms = ArbStudio.load_waveforms(channels=[1,2,3])
#     sequences = ArbStudio.load_sequence(channels=[1,2,3])


if __name__ == "__main__":
    from meta_instruments import HackInstrument

    def f(x):
        print('hi')
        return x + 1

    hack_instrument = HackInstrument.HackInstrument(name='Hack_instrument')

    hack_instrument.set_function(f)
    # import qcodes.instrument_drivers.lecroy.ArbStudio1104 as ArbStudio_driver
    #
    # dll_path = os.path.join(os.getcwd(), 'lecroy_driver\\Library\\ArbStudioSDK.dll')
    # ArbStudio = ArbStudio_driver.ArbStudio1104('ArbStudio', dll_path, server_name=None)
    #
    # configure_ArbStudio(ArbStudio, voltages = {'empty': -1.5, 'load': 1.5, 'read': 0},
    #                     channel_factors=[1, -1.5, 1]
    #                     )