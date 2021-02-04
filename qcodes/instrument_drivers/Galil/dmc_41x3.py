"""
This file holds the QCoDeS driver for the Galil DMC-41x3 motor controllers,
colloquially known as the "stepper motors".
"""
from typing import Any, Dict, Optional, List
import numpy as np

from qcodes import Instrument
from qcodes.utils.validators import Enum, Ints

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
                 **kwargs: Any) -> None:
        super().__init__(name=name, address=address, **kwargs)

        self.add_parameter("move_a",
                           set_cmd=self._move_motor_a,
                           vals=Ints(-2147483648, 2147483647),
                           units="quadrature counts",
                           docstring="moves motor A along x-axis. negative "
                                     "value indicates movement along -x axis.")

        self.add_parameter("move_b",
                           set_cmd=self._move_motor_b,
                           vals=Ints(-2147483648, 2147483647),
                           units="quadrature counts",
                           docstring="moves motor B along y-axis. negative "
                                     "value indicates movement along -y axis.")

        self.add_parameter("move_c",
                           set_cmd=self._move_motor_c,
                           vals=Ints(-2147483648, 2147483647),
                           units="quadrature counts",
                           docstring="moves motor C along z-axis. negative "
                                     "value indicates movement along -z axis.")

        self.add_parameter("position_format_decimals",
                           set_cmd="PF 10.{}",
                           vals=Ints(0, 4),
                           docstring="sets number of decimals in the format "
                                     "of the position")

        self.add_parameter("absolute_position",
                           get_cmd=self._get_absolute_position,
                           units="quadrature counts",
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

        self.add_parameter("begin",
                           set_cmd="BG {}",
                           vals=Enum("A", "B", "C", "S"),
                           docstring="begins the specified motor or sequence "
                                     "motion")

        self.add_parameter("servo_at_motor",
                           set_cmd="SH {}",
                           vals=Enum("A", "B", "C"),
                           docstring="servo at the specified motor"
                           )

        self.add_parameter("after_motion",
                           set_cmd="AM {}",
                           vals=Enum("A", "B", "C", "S"),
                           docstring="wait till motion of given motor or "
                                     "sequence finishes")

        self.add_parameter("wait",
                           set_cmd="WT {}",
                           units="ms",
                           vals=Enum(np.linespace(2, 2147483646, 2)),
                           docstring="controller will wait for the amount of "
                                     "time specified before executing the next "
                                     "command")

        self.add_parameter("coordinate_system",
                           get_cmd="CA ?",
                           get_parser=self._parse_coordinate_system_active,
                           set_cmd="CA {}",
                           vals=Enum("S", "T"),
                           docstring="sets coordinate system for the motion")

        self.add_parameter("clear_sequence",
                           set_cmd="CS {}",
                           vals=Enum("S", "T"),
                           docstring="clears vectors specified in the given "
                                     "coordinate system")

        self.add_parameter("vector_mode",
                           set_cmd="VM {}",
                           vals=Enum("AB", "BC", "AC"),
                           docstring="sets plane of motion for the motors")

        self.add_parameter("vector_position",
                           set_cmd="VP {},{}", #make group param
                           vals=Ints(-2147483648, 2147483647),
                           units="quadrature counts",
                           docstring="can set initial and final vector "
                                     "positions for the motion")

        self.add_parameter("vector_acceleration",
                           get_cmd="VA ?",
                           get_parser=int,
                           set_cmd="VA {}",
                           vals=Enum(np.linespace(1024, 1073740800, 1024)),
                           units="counts/sec2",
                           docstring="sets and gets the defined vector's "
                                     "acceleration")

        self.add_parameter("vector_deceleration",
                           get_cmd="VD ?",
                           get_parser=int,
                           set_cmd="VD {}",
                           vals=Enum(np.linspace(1024, 1073740800, 1024)),
                           units="counts/sec2",
                           docstring="sets and gets the defined vector's "
                                     "deceleration")

        self.add_parameter("vector_speed",
                           get_cmd="VS ?",
                           get_parser=int,
                           set_cmd="VS {}",
                           vals=Enum(np.linespace(2, 15000000, 2)),
                           units="counts/sec",
                           docstring="sets and gets defined vector's speed")

        self.add_parameter("half_circle_move_of_radius",
                           set_cmd="CR {},0,180",
                           vals=Ints(10, 6000000),
                           units="quadrature counts",
                           docstring="defines half circle move in the "
                                     "set vector mode")

        self.add_parameter("vector_seq_end",
                           set_cmd="VE",
                           docstring="indicates to the controller that the end"
                                     " of the vector is coming up. is "
                                     "required to exit the vector mode "
                                     "gracefully")
        self.connect_message()

    def _move_motor_a(self, val: int) -> None:
        """
        moves motor A to the given amount from the current position
        """
        self.motor_off("A")
        self.servo_at_motor("A")
        self.write(f"PRA={val}")
        self.write("SPA=1000")
        self.write("ACA=500000")
        self.write("DCA=500000")
        self.begin("A")

    def _move_motor_b(self, val: int) -> None:
        """
        moves motor B to the given amount from the current position
        """
        self.motor_off("B")
        self.servo_at_motor("B")
        self.write(f"PRB={val}")
        self.write("SPB=1000")
        self.write("ACB=500000")
        self.write("DCB=500000")
        self.begin("B")

    def _move_motor_c(self, val: int) -> None:
        """
        moves motor C to the given amount from the current position
        """
        self.motor_off("C")
        self.servo_at_motor("C")
        self.write(f"PRC={val}")
        self.write("SPC=1000")
        self.write("ACC=500000")
        self.write("DCC=500000")
        self.begin("C")

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

    def _get_absolute_position(self) -> Dict[str, int]:
        """
        gets absolution position of the motors from the defined origin
        """
        result = dict()
        data = self.ask("PA ?,?,?").split(" ")
        result["A"] = int(data[0][:-1])
        result["B"] = int(data[1][:-1])
        result["C"] = int(data[2])

        return result

    @staticmethod
    def _parse_coordinate_system_active(val: str) -> str:
        """
        parses the the current active coordinate system
        """
        if int(val):
            return "T"
        else:
            return "S"

    def end_program(self) -> None:
        """
        ends the program
        """
        self.write("EN")

    def define_position_as_origin(self) -> None:
        """
        defines current motors position as origin
        """
        self.write("DP 0,0,0")

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

    def home(self) -> None:
        """
         performs a three stage homing sequence for servo systems and a two
         stage sequence for stepper motors.

         Step One. Servos and Steppers
            - During the first stage of the homing sequence, the motor moves at
            the user-programmed speed until detecting a transition on the
            homing input for that axis. The speed for step one is set with the
            SP command.

            - The direction for this first stage is determined by the
            initial state of the homing input. The state of the homing input
            can be configured using the second field of the CN command.

            - Once the homing input changes state, the motor decelerates to a
            stop.

        Step Two. Servos and Steppers
            - At the second stage, the motor changes directions and
            approaches the transition again at the speed set with the
            HV command. When the transition is detected, the motor is stopped
            instantaneously.

        Step Three. Servos only
            - At the third stage, the motor moves in the positive direction
            at the speed set with the HV command until it detects an index
            pulse via latch from the encoder. It returns to the latched
            position and defines it as position 0.
        """
        # setup for homing
        self.write("SP 2000,2000,2000")
        self.write("CN ,-1")
        self.write("HV 256,256,256")

        # home command
        self.write("HM")

        # begin motion
        self.write("BG")

        # wait for motion to finish
        self.write("AM")

    def error_magnitude(self) -> Dict[str, int]:
        """
        gives the magnitude of error, in drive step counts, for axes in
        Stepper Position Maintenance mode.

        a step count is directly proportional to the micro-stepping
        resolution of the stepper drive.
        """
        data = self.ask("QS").split(",")
        return {"A": int(data[0]), "B": int(data[1]), "C": int(data[2])}

    def _setup_spm(self) -> None:
        """
        sets up for Stepper Position Maintenance (SPM) mode
        """
        self.write("OE 1,1,1")   # Set the profiler to stop axis upon error
        self.write("KS 16,16,16")  # Set step smoothing
        self.write("MT -2,-2,-2")  # Motor type set to stepper
        self.write("YA 1,1,1")     # Step resolution of the drive
        self.write("YB 200,200,200")   # Motor resolution (full steps per revolution)
        self.write("YC 4000,4000,4000")  # Encoder resolution (counts per revolution)

    def enable_stepper_position_maintenance_mode(self, motor: str) -> None:
        """
        enables Stepper Position Maintenance mode and allows for error
        correction when error happens
        """
        cmd = "YS"
        if motor == "A":
            cmd = cmd + " 1"
        elif motor == "B":
            cmd = cmd + " ,1"
        else:
            cmd = cmd + " ,,1"

        self._setup_spm()
        self.servo_at_motor(motor)  # Enable axis
        self.wait(50)  # Allow slight settle time
        self.write(cmd)


class Arm:

    def __init__(self, driver: DMC4133, chip_design: str):
        self.driver = driver
        self.chip_design = chip_design
        self.load_chip_design(self.chip_design)

    def load_chip_design(self, filename: str) -> None:
        """
        loads chip design features such as width and height of the chip,
        pads dimensions and intra-pads measurements
        """
        self._chip_length: float
        self._chip_width: float
        self._rows: int
        self._num_terminals_in_row: int
        self._terminal_length: float
        self._terminal_width: float
        self._inter_terminal_distance_for_adjacent_rows: float

    def move_to_next_row(self) -> int:
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
