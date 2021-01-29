"""
This file holds the QCoDeS driver for the Galil DMC-41x3 motor controllers,
colloquially known as the "stepper motors".
"""
from typing import Any, Dict, Optional, List

from qcodes import Instrument
from qcodes.utils.validators import Enum, Numbers, Ints

try:
    import gclib
except ImportError as e:
    raise ImportError(
        "Cannot find gclib library. Download gclib installer from "
        "https://www.galil.com/sw/pub/all/rn/gclib.html and install Galil "
        "motion controller software for your OS. Afterwards go "
        "to https://www.galil.com/sw/pub/all/doc/gclib/html/python.html and "
        "follow instruction to be able to import gclib package in your "
        "environment.") from e


class GalilInstrument(Instrument):
    """
    Base class for Galil Motion Controller drivers
    """
    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.g = gclib.py()
        self.address = address

    def get_idn(self) -> Dict[str, Optional[str]]:
        """
        Get Galil motion controller hardware information
        """
        self.log.info('Listening for controllers requesting IP addresses...')
        ip_requests = self.g.GIpRequests()
        if len(ip_requests) != 1:
            raise RuntimeError("Multiple or No controllers connected!")

        instrument = list(ip_requests.keys())[0]
        self.log.info(instrument + " at mac" + ip_requests[instrument])

        self.log.info("Assigning " + self.address +
                      " to mac" + ip_requests[instrument])
        self.g.GAssign(self.address, ip_requests[instrument])
        self.g.GOpen(self.address + ' --direct')

        data = self.g.GInfo().split(" ")
        idparts: List[Optional[str]] = ["Galil Motion Control, Inc.",
                                        data[1], data[4], data[3][:-1]]

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def write_raw(self, cmd: str) -> None:
        """
        Write for Galil motion controller
        """
        self.g.GCommand(cmd+"\r")

    def ask_raw(self, cmd: str) -> str:
        """
        Asks/Reads data from Galil motion controller
        """
        return self.g.GCommand(cmd+"\r")

    def timeout(self, val: float) -> None:
        """
        Sets timeout for the instrument

        Args:
            val: time in seconds
        """
        if val < 0.001:
            raise RuntimeError("Timeout can not be less than 0.001s")

        self.g.GTimeout(val*1000)

    def close(self) -> None:
        """
        Close connection to the instrument
        """
        self.g.GClose()


class DMC4133(GalilInstrument):
    """
    Driver for Galil DMC-4133 Motor Controller
    """

    def __init__(self,
                 name: str,
                 address: str,
                 chip_design: str,
                 **kwargs: Any) -> None:
        super().__init__(name=name, address=address, **kwargs)
        self.chip_design = chip_design
        self.load_chip_design(self.chip_design)

        self.add_parameter("move_a",
                           set_cmd=self._move_motor_a,
                           units="microns",
                           vals=Numbers(-107374182.4, 107374182.35),  # 0.05 res
                           docstring="moves motor a along x-axis. negative "
                                     "value indicates movement along -x axis.")

        self.add_parameter("move_b",
                           set_cmd=self._move_motor_b,
                           units="microns",
                           vals=Numbers(-107374182.4, 107374182.35),
                           docstring="moves motor b along y-axis. negative "
                                     "value indicates movement along -y axis.")

        self.add_parameter("move_c",
                           set_cmd=self._move_motor_c,
                           units="microns",
                           vals=Numbers(-107374182.4, 107374182.35),
                           docstring="moves motor c along z-axis. negative "
                                     "value indicates movement along -z axis.")

        self.add_parameter("position_format_decimals",
                           set_cmd="PF 10.{}",
                           vals=Ints(0, 4),
                           docstring="sets number of decimals in the format "
                                     "of the position")

        self.add_parameter("absolute_position",
                           get_cmd=self._get_absolute_position,
                           units="microns",
                           docstring="gets absolute position of the motors "
                                     "from the set origin")

        self.add_parameter("motor_off",
                           get_cmd="MG _MO",
                           get_parser=self._motor_on_off_status,
                           set_cmd="MO {}",
                           vals=Enum("A", "B", "C"),
                           docstring="turns given motors off and when called "
                                     "without argument tells the status of "
                                     "motors")

        self.add_parameter("begin_motor",
                           set_cmd="BG {}",
                           vals=Enum("A", "B", "C"),
                           docstring="begins the specified motor motion")

        self.add_parameter("servo_at_motor",
                           set_cmd="SH {}",
                           vals=Enum("A", "B", "C"),
                           docstring="servo at the specified motor"
                           )

        self.add_parameter("after_motion_of_motor",
                           set_cmd="AM {}",
                           vals=Enum("A", "B", "C"),
                           docstring="wait till motion of given motor finishes")

        self.add_parameter("wait",
                           set_cmd="WT {}",
                           units="ms",
                           vals=Ints(2, 2147483646),  # resolution is 2 find how
                           docstring="controller will wait for the amount of "
                                     "time specified before executing the next "
                                     "command")

        self.connect_message()

    def _move_motor_a(self, val: int) -> None:
        """
        this method converts the given distance in microns into quadrature
        counts and moves motor A to that amount from the current position

        note: 50 microns equals 1000 quadrature counts
        """
        self.motor_off("A")
        self.servo_at_motor("A")
        self.write(f"PRA={val*20}")
        self.write("SPA=1000")
        self.write("ACA=500000")
        self.write("DCA=500000")
        self.begin_motor("A")

    def _move_motor_b(self, val: int) -> None:
        """
        this method converts the given distance in microns into quadrature
        counts and moves motor B to that amount from the current position

        note: 50 microns equals 1000 quadrature counts
        """
        self.motor_off("B")
        self.servo_at_motor("B")
        self.write(f"PRB={val*20}")
        self.write("SPB=1000")
        self.write("ACB=500000")
        self.write("DCB=500000")
        self.begin_motor("B")

    def _move_motor_c(self, val: int) -> None:
        """
        this method converts the given distance in microns into quadrature
        counts and moves motor C to that amount from the current position

        note: 50 microns equals 1000 quadrature counts
        """
        self.motor_off("C")
        self.servo_at_motor("C")
        self.write(f"PRC={val*20}")
        self.write("SPC=1000")
        self.write("ACC=500000")
        self.write("DCC=500000")
        self.begin_motor("C")

    @staticmethod
    def _motor_on_off_status(val: str) -> Dict[str, str]:
        """
        motor on off status parser
        """
        result = dict()
        data = val.split(" ")

        if int(data[0][:-1]) == 1:
            result["A"] = "OFF"
        else:
            result["A"] = "ON"

        if int(data[1][:-1]) == 1:
            result["B"] = "OFF"
        else:
            result["B"] = "ON"

        if int(data[2]) == 1:
            result["C"] = "OFF"
        else:
            result["C"] = "ON"

        return result

    def _get_absolute_position(self) -> Dict[str, float]:
        """
        gets absolution position of the motors from the defined origin
        """
        result = dict()
        data = self.ask("PA ?,?,?").split(" ")
        result["A"] = int(data[0][:-1])/20
        result["B"] = int(data[1][:-1])/20
        result["C"] = int(data[2])/20

        return result

    def _define_position_as_origin(self) -> None:
        """
        defines current motors position as origin
        """
        self.write("DP 0,0,0")

    def _move_to_next_row(self) -> int:
        """
        moves motors to next row of pads
        """
        pass

    def set_begin_position(self) -> None:
        """
        sets first row of pads in chip as begin position
        """
        pass

    def set_end_position(self) -> None:
        """
        sets last row of pads in chip as end position
        """
        pass

    def begin_motion(self) -> str:
        """
        begins motion of motors after setup
        """
        pass

    def load_chip_design(self, filename: str) -> None:
        """
        loads chip design features such as width and height of the chip,
        pads dimensions and intra-pads measurements
        """
        pass

    def tell_error(self) -> str:
        """
        reads error
        """
        return self.ask("TC1")

    def stop(self) -> None:
        """
        stop the motion of all motors
        """
        self.write("ST")

    def abort(self) -> None:
        """
        aborts motion and the program operation
        """
        self.write("AB")
