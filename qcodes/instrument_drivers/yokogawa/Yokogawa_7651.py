from qcodes import VisaInstrument
from qcodes.utils.validators import Numbers, Ints

def voltage_parser(instrument_response):
    if instrument_response.startswith('NDCV'):
        return float(instrument_response.lstrip('NDCV'))
    else:
        raise Exception('Wrong response: {0}'.format(instrument_response))

class Yokogawa_7651(VisaInstrument):
    """This is the code for the Yokogawa 7651 voltage source."""
    def __init__(self, name, address, reset=False, **kwargs):
        super().__init__(name, address, terminator='\r\n', device_clear=False,
            **kwargs)

        self.add_parameter(name='voltage',
                           label='Voltage',
                           unit='V',
                           get_cmd='OD',
                           set_cmd='S{:.5E}\r\nE',
                           get_parser=voltage_parser,
                           vals=Numbers(min_value=-30.0, max_value=30.0),
                           post_delay=0.1)
        self.add_parameter(name='voltage_limit',
                           label='Voltage limit',
                           unit='V',
                           set_cmd='LV{:d}',
                           post_delay=0.1,
                           vals=Ints(min_value=1, max_value=30))
        self.add_parameter(name='panel_settings',
                           label='Panel settings',
                           get_cmd=self._get_panel_settings,
                           )
        self.add_parameter(name='status',
                           label='current status',
                           get_cmd='OC')


        self.add_function('enable_output', call_cmd='O1E')
        self.add_function('disable_output', call_cmd='O0E')

    def _get_panel_settings(self):
        """Read the panel settings.
        
        Notes:
            The instrument sends five lines:
                1 : MDL7651REV1.05
                2 :
                3 :
                4 : limit settings
                5 : END
        """
        settings = []
        response = self.ask('OS')
        settings.append(response)
        for _ in range(4):
            response = self.visa_handle.read()
            settings.append(response)
        if response != 'END':
            raise Exception('Settings query response unexpected.')
        return settings
        

