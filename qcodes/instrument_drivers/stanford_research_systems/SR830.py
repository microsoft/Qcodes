from qcodes import VisaInstrument
from qcodes.utils.validators import Numbers, Ints

class Stanford_Research_Systems_SR830(VisaInstrument):
    """
    Driver for the Stanford Research Systems SR830 lock-in amplifier.
    """
    def __init__(self, name, address, reset=False, **kwargs):
        super().__init__(name, address, **kwargs)

        # Reference and phase
        self.add_parameter('phase',
                           get_cmd='PHAS?',
                           get_parser=float,
                           set_cmd='PHAS {}',
                           units='deg',
                           vals=Numbers(min_value=-360, max_value=729.99))

        self.add_parameter('reference_source',
                           get_cmd='FMOD?',
                           set_cmd='FMOD {}',
                           val_mapping={
                               'external': '0\n',
                               'internal': '1\n',
                           })

        self.add_parameter('frequency',
                           get_cmd='FREQ?',
                           get_parser=float,
                           set_cmd='FREQ {:e}',
                           units='Hz',
                           vals=Numbers(min_value=1e-3, max_value=102e3))

        self.add_parameter('ext_trigger',
                           get_cmd='RSLP?',
                           set_cmd='RSLP {}',
                           val_mapping={
                               'sine': '0\n',
                               'ttl rising': '1\n',
                               'ttl falling': '2\n',
                           })

        self.add_parameter('harmonic',
                           get_cmd='HARM?',
                           get_parser=int,
                           set_cmd='HARM {}',
                           vals=Ints(min_value=1, max_value=19999))

        self.add_parameter('amplitude',
                           get_cmd='SLVL?',
                           get_parser=float,
                           set_cmd='SLVL {}',
                           units='V',
                           vals=Numbers(min_value=0.004, max_value=5.000))

        # Input and filter
        self.add_parameter('input_config',
                           get_cmd='ISRC?',
                           set_cmd='ISRC {}',
                           val_mapping={
                               'a': '0\n',
                               'a-b': '1\n',
                               '1 mohm': '2\n',
                               '100 mohm': '3\n',
                           })

        self.add_parameter('input_shield',
                           get_cmd='IGND?',
                           set_cmd='IGND {}',
                           val_mapping={
                               'float': '0\n',
                               'ground': '1\n',
                           })

        self.add_parameter('input_coupling',
                           get_cmd='ICPL?',
                           set_cmd='ICPL {}',
                           val_mapping={
                               'ac': '0\n',
                               'dc': '1\n',
                           })

        self.add_parameter('notch_filter',
                           get_cmd='ILIN?',
                           set_cmd='ILIN {}',
                           val_mapping={
                               'out': '0\n',
                               'line in': '1\n',
                               '2x line in': '2\n',
                               'both in': '3\n',
                           })

        # Gain and time constant
        self.add_parameter('sensitivity',
                           get_cmd='SENS?',
                           set_cmd='SENS {}',
                           val_mapping={
                               "2 nv": '0\n',
                               "5 nv": '1\n',
                               "10 nv": '2\n',
                               "20 nv": '3\n',
                               "50 nv": '4\n',
                               "100 nv": '5\n',
                               "200 nv": '6\n',
                               "500 nv": '7\n',
                               "1 uv": '8\n',
                               "2 uv": '9\n',
                               "5 uv": '10\n',
                               "10 uv": '11\n',
                               "20 uv": '12\n',
                               "50 uv": '13\n',
                               "100 uv": '14\n',
                               "200 uv": '15\n',
                               "500 uv": '16\n',
                               "1 mv": '17\n',
                               "2 mv": '18\n',
                               "5 mv": '19\n',
                               "10 mv": '20\n',
                               "20 mv": '21\n',
                               "50 mv": '22\n',
                               "100 mv": '23\n',
                               "200 mv": '24\n',
                               "500 mv": '25\n',
                               "1 v": '26\n',
                           })

        self.add_parameter('reserve_mode',
                           get_cmd='RMOD?',
                           set_cmd='RMOD {}',
                           val_mapping={
                               'high reserve': '0\n',
                               'normal': '1\n',
                               'low noise': '2\n',
                           })

        self.add_parameter('time_constant',
                           get_cmd='OFLT?',
                           set_cmd='OFLT {}',
                           val_mapping={
                               "10 us": '0\n',
                               "30 us": '1\n',
                               "100 us": '2\n',
                               "300 us": '3\n',
                               "1 ms": '4\n',
                               "3 ms": '5\n',
                               "10 ms": '6\n',
                               "30 ms": '7\n',
                               "100 ms": '8\n',
                               "300 ms": '9\n',
                               "1 s": '10\n',
                               "3 s": '11\n',
                               "10 s": '12\n',
                               "30 s": '13\n',
                               "100 s": '14\n',
                               "300 s": '15\n',
                               "1 ks": '16\n',
                               "3 ks": '17\n',
                               "10 ks": '18\n',
                               "30 ks": '19\n',
                           })

        self.add_parameter('filter_slope',
                           get_cmd='OFSL?',
                           set_cmd='OFSL {}',
                           val_mapping={
                               '6 db/oct': '0\n',
                               '12 db/oct': '1\n',
                               '18 db/oct': '2\n',
                               '24 db/oct': '3\n',
                           })

        self.add_parameter('sync_filter',
                           get_cmd='SYNC?',
                           set_cmd='SYNC {}',
                           val_mapping={
                               'off': '0\n',
                               'on': '1\n',
                           })

        # Display and output

        # Aux input/output
        for i in [0, 1, 2, 3]:
            self.add_parameter('aux_in{}'.format(i),
                               get_cmd='OAUX? {}'.format(i),
                               get_parser=float,
                               units='V')

            self.add_parameter('aux_out{}'.format(i),
                               get_cmd='AUXV? {}'.format(i),
                               get_parser=float,
                               set_cmd='AUXV {0}, {{}}'.format(i),
                               units='V')

        # Setup
        self.add_parameter('output_interface',
                           get_cmd='OUTX?',
                           set_cmd='OUTX {}',
                           val_mapping={
                               'rs232': '0\n',
                               'gpib': '1\n',
                           })

        # Auto functions
        self.add_function('auto_gain', call_cmd='AGAN')
        self.add_function('auto_reserve', call_cmd='ARSV')
        self.add_function('auto_phase', call_cmd='APHS')

        # Data storage

        # Data transfer
        self.add_parameter('X', 
                           get_cmd='OUTP?1', 
                           get_parser=float, 
                           units='V')

        self.add_parameter('Y', 
                           get_cmd='OUTP?2', 
                           get_parser=float, 
                           units='V')

        self.add_parameter('R', 
                           get_cmd='OUTP?3', 
                           get_parser=float, 
                           units='V')

        self.add_parameter('P', 
                           get_cmd='OUTP?4', 
                           get_parser=float, 
                           units='deg')

        # Interface
        self.add_function('reset', call_cmd='*RST')

        self.add_function('disable_front_panel', call_cmd='OVRM 0')
        self.add_function('enable_front_panel', call_cmd='OVRM 1')