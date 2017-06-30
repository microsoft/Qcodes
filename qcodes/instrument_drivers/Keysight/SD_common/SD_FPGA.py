from .SD_Module import *


class SD_FPGA(SD_Module):
    """
        This is the general SD_FPGA driver class to access FPGA communication common among all PXIe-based
        Keysight M32xxA and M33xxA cards.

        This driver makes use of the Python library provided by Keysight as part of the SD1 Software package (v.2.01.00).
    """

    def __init__(self, name, chassis, slot, **kwargs):
        super().__init__(name, chassis, slot, **kwargs)

        # Open the device, using the specified chassis and slot number
        FPGA_name = self.SD_module.getProductNameBySlot(chassis, slot) + '_FPGA'
        if isinstance(FPGA_name, str):
            result_code = self.awg.openWithSlot(FPGA_name, chassis, slot)
            if result_code <= 0:
                raise Exception('Could not open FPGA '
                                'error code {}'.format(result_code))
        else:
            raise Exception('No FPGA found at '
                            'chassis {}, slot {}'.format(chassis, slot))

    def start(self):
        """ Starts the currently loaded FPGA firmware. """
        raise NotImplementedError('start() must be defined per your FPGA design.')

    def stop(self):
        """ Stops the currently loaded FPGA firmware. """
        raise NotImplementedError('stop() must be defined per your FPGA design.')

    def reset(self):
        """ Resets the currently loaded FPGA firmware. """
        raise NotImplementedError('reset() must be defined per your FPGA design.')

    def get_fpga_pc_port(self, port, data_size, address, address_mode, access_mode,
                         verbose=False):
        """
        Reads data at the PCport FPGA Block

        Args:
            port (int): PCport number
            data_size (int): number of 32-bit words to read (maximum is 128 words)
            address (int): address that will appear at the PCport interface
            address_mode (int): 0 (auto-increment) or 1 (fixed)
            access_mode (int): 0 (non-DMA) or 1 (DMA)
        """
        data = self.SD_module.FPGAreadPCport(port, data_size, address, address_mode,
                                             access_mode)
        value_name = 'data at PCport {}'.format(port)
        return result_parser(data, value_name, verbose)


    def set_fpga_pc_port(self, port, data, address, address_mode, access_mode,
                         verbose=False):
        """
        Writes data at the PCport FPGA Block

        Args:
            port (int): PCport number
            data (array): array of integers containing the data
            address (int): address that wil appear at the PCport interface
            address_mode (int): 0 (auto-increment) or 1 (fixed)
            access_mode (int): 0 (non-DMA) or 1 (DMA)
        """
        result = self.SD_module.FPGAwritePCport(port, data, address, address_mode,
                                                access_mode)
        value_name = 'set fpga PCport {} to data:{}, address:{}, address_mode:{}, access_mode:{}' \
            .format(port, data, address, address_mode, access_mode)
        return result_parser(result, value_name, verbose)
