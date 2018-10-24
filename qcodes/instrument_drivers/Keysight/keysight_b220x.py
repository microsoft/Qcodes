from qcodes import VisaInstrument, validators


class KeysightB220X(VisaInstrument):
    """
    QCodes driver for B2200 / B2201 switch matrix
    """

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\r', **kwargs)

        self.add_function(name='connect',
                          call_cmd=self._connect,
                          args=[validators.Ints(1, 14), validators.Ints(1, 48)])

        self.add_function(name='disconnect',
                          call_cmd=self._connect,
                          args=[validators.Ints(1, 14), validators.Ints(1, 48)])

        self.add_function(name='disconnect_all',
                          call_cmd=':OPEN:CARD 0')

        self.add_parameter(name='connection_rule',
                           get_cmd=':CONN:RULE? 0',
                           set_cmd=':CONN:RULE 0,{}',
                           val_mapping={'free': 'FREE',
                                        'single': 'SROU'})

        self.add_parameter(name='connection_sequence',
                           get_cmd=':CONN:SEQ? 0',
                           set_cmd=':CONN:SEQ 0,{}',
                           val_mapping={'none': 'NSEQ',
                                       'bbm': 'BBM',
                                       'mbb': 'MBBR'})

        self.add_function(name='bias_disable_all',
                          call_cmd=':BIAS:CHAN:DIS:CARD 0')

        self.add_function(name='bias_enable_all',
                          call_cmd=':BIAS:CHAN:ENAB:CARD 0')

        self.add_function(name='bias_enable_channel',
                          call_cmd=':BIAS:CHAN:ENAB (@001{:02d})',
                          args=[validators.Ints(1, 48), ])

        self.add_function(name='bias_disable_channel',
                          call_cmd=':BIAS:CHAN:DIS (@001{:02d})',
                          args=[validators.Ints(1, 48), ])

        self.add_parameter(name='bias_input_port',
                           get_cmd=':BIAS:PORT? 0',
                           set_cmd=':BIAS:PORT 0,{}',
                           vals=validators.MultiType(validators.Ints(1, 14),
                                                     validators.Enum(-1))
                           )

    def _connect(self, channel1, channel2):
        self.write(":CLOS (@{card:01d}{ch1:02d}{ch2:02d})".format(card=0,
                                                                ch1=channel1,
                                                                ch2=channel2))

    def _disconnect(self, channel1, channel2):
        self.write(":OPEN (@{card:01d}{ch1:02d}{ch2:02d})".format(card=0,
                                                                ch1=channel1,
                                                                ch2=channel2))




