from qcodes import Instrument


class Thorlabs_PRM1Z8(Instrument):
    """
    Instrument driver for the Thorlabs PRMZ1Z8 polarizer wheel.
    Args:
        name (str): Instrument name.
        device_id (int): ID for the desired polarizer wheel.
        apt (Thorlabs_APT): Thorlabs APT server.
    Attributes:
        apt (Thorlabs_APT): Thorlabs APT server.
        serial_number (int): Serial number of the polarizer wheel.
        model (str): Model description.
        version (str): Firmware version.
    """

    def __init__(self, name, device_id, apt, **kwargs):

        super().__init__(name, **kwargs)

        # save APT server link
        self.apt = apt

        # initialization
        self.serial_number = self.apt.get_hw_serial_num_ex(31, device_id)
        self.apt.init_hw_device(self.serial_number)
        self.model = self.apt.get_hw_info(self.serial_number)[0].decode("utf-8")
        self.version = self.apt.get_hw_info(self.serial_number)[1].decode("utf-8")

        # add parameters
        self.add_parameter('position',
                           get_cmd=self.get_position,
                           set_cmd=self.set_position,
                           unit=u"\u00b0",
                           label='position')

        # print connect message
        self.connect_message(idn_param='IDN')

    # get methods
    def get_idn(self):
        return {'vendor': 'Thorlabs', 'model': self.model,
                'firmware': self.version, 'serial': self.serial_number}

    def get_position(self):
        return self.apt.mot_get_position(self.serial_number)

    # set methods
    def set_position(self, position):
        self.apt.mot_move_absolute_ex(self.serial_number, position, True)
