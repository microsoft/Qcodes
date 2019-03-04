from qcodes import Instrument


class Thorlabs_MFF10x(Instrument):
    """
    Instrument driver for the Thorlabs MFF10x mirror flipper.
    Args:
        name (str): Instrument name.
        device_id (int): ID for the desired mirror flipper.
        apt (Thorlabs_APT): Thorlabs APT server.
    Attributes:
        apt (Thorlabs_APT): Thorlabs APT server.
        serial_number (int): Serial number of the mirror flipper.
        model (str): Model description.
        version (str): Firmware version.
    """

    def __init__(self, name, device_id, apt, **kwargs):

        super().__init__(name, **kwargs)

        # save APT server link
        self.apt = apt

        # initialization
        self.serial_number = self.apt.get_hw_serial_num_ex(48, device_id)
        self.apt.init_hw_device(self.serial_number)
        self.model = self.apt.get_hw_info(self.serial_number)[0].decode("utf-8")
        self.version = self.apt.get_hw_info(self.serial_number)[1].decode("utf-8")

        # add parameters
        self.add_parameter('position',
                           get_cmd=self.get_position,
                           set_cmd=self.set_position,
                           get_parser=int,
                           label='position')

        # print connect message
        self.connect_message(idn_param='IDN')

    # get methods
    def get_idn(self):
        return {'vendor': 'Thorlabs', 'model': self.model,
                'firmware': self.version, 'serial': self.serial_number}

    def get_position(self):
        status_bits = bin(self.apt.mot_get_status_bits(self.serial_number) & 0xffffffff)
        return status_bits[-1]

    # set methods
    def set_position(self, position):
        self.apt.mot_move_jog(self.serial_number, position+1, False)
