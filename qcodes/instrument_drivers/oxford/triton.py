import configparser
from qcodes import IPInstrument


class Triton(IPInstrument):
    """
    Triton Driver
    TODO: fetch registry directly from fridge-computer

    Comment from Merlin:
        I had to change the IP instrument, somehow the port does not get
        transferred to the socket connection
        And I had other problems with the _connect method,
          check those changes :)
        Further there was a problem with the EnsureConnection class
    """

    def __init__(self, name, address=None, port=None, terminator='\r\n',
                 tmpfile=None, **kwargs):
        super().__init__(name, address=address, port=port,
                         terminator=terminator, **kwargs)

        self.add_parameter(name='time',
                           label='System Time',
                           units='',
                           get_cmd='READ:SYS:TIME',
                           get_parser=self._parse_time)

        self.add_parameter(name='action',
                           label='Current action',
                           units='',
                           get_cmd='READ:SYS:DR:ACTN',
                           get_parser=self._parse_action)

        self.add_parameter(name='status',
                           label='Status',
                           units='',
                           get_cmd='READ:SYS:DR:STATUS',
                           get_parser=self._parse_status)

        self.chan_alias = {}
        self.chan_temps = {}
        if tmpfile is not None:
            self._get_temp_channels(tmpfile)
        self._get_pressure_channels()
        self._get_named_channels()

        self.connect_message()

    def _get_named_channels(self):
        allchans = self.ask('READ:SYS:DR:CHAN')
        allchans = allchans.replace('STAT:SYS:DR:CHAN:', '', 1).split(':')
        for ch in allchans:
            msg = 'READ:SYS:DR:CHAN:%s' % ch
            rep = self.ask(msg)
            if 'INVALID' not in rep:
                alias, chan = rep.split(':')[-2:]
                self.chan_alias[alias] = chan
                self.add_parameter(name=alias,
                                   units='K',
                                   get_cmd='READ:DEV:%s:TEMP:SIG:TEMP' % chan,
                                   get_parser=self._parse_temp)

    def _get_pressure_channels(self):
        for i in range(1, 7):
            chan = 'P%d' % i
            self.add_parameter(name=chan,
                               units='bar',
                               get_cmd='READ:DEV:%s:PRES:SIG:PRES' % chan,
                               get_parser=self._parse_pres)

    def _get_temp_channels(self, file):
        config = configparser.ConfigParser()
        with open(file, 'r') as f:
            next(f)
            config.read_file(f)

        for section in config.sections():
            options = config.options(section)
            namestr = '"m_lpszname"'
            if namestr in options:
                chan = 'T'+section.split('\\')[-1].split('[')[-1]
                name = config.get(section, '"m_lpszname"').strip("\"")
                self.chan_temps[chan] = {'name': name, 'value': None}
                self.add_parameter(name=chan,
                                   units='K',
                                   get_cmd='READ:DEV:%s:TEMP:SIG:TEMP' % chan,
                                   get_parser=self._parse_temp)

    def _parse_action(self, msg):
        action = msg[17:]
        if action == 'PCL':
            action = 'Precooling'
        elif action == 'EPCL':
            action = 'Empty precool loop'
        elif action == 'COND':
            action = 'Condensing'
        elif action == 'NONE':
            if self.MC.get() < 2:
                action = 'Circulating'
            else:
                action = 'Idle'
        elif action == 'COLL':
            action = 'Collecting mixture'
        else:
            action = 'Unknown'
        return action

    def _parse_status(self, msg):
        return msg[19:]

    def _parse_time(self, msg):
        return msg[14:]

    def _parse_temp(self, msg):
        if 'NOT_FOUND' in msg:
            return None
        return float(msg.split('SIG:TEMP:')[-1].strip('K'))

    def _parse_pres(self, msg):
        if 'NOT_FOUND' in msg:
            return None
        return float(msg.split('SIG:PRES:')[-1].strip('mB'))*1e3

    def _recv(self):
        return super()._recv().rstrip()
