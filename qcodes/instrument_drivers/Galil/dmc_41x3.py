"""
This file holds the QCoDeS driver for the Galil DMC-41x3 motor controllers,
colloquially known as the "stepper motors".
"""
from typing import Any, Dict, Optional, List
import numpy as np

from qcodes import Instrument, InstrumentChannel
from qcodes.utils.validators import Enum, Ints, Union

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


class GalilMotionController(Instrument):
    """
    Base class for Galil Motion Controller drivers
    """
    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.g = gclib.py()
        self.address = address
        self.open()

    def open(self) -> None:
        """
        Open connection to Galil motion controller
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

    def get_idn(self) -> Dict[str, Optional[str]]:
        """
        Get Galil motion controller hardware information
        """
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


class VectorMode(InstrumentChannel):
    """
    Class to control motors independently
    """

    def __init__(self,
                 parent: "DMC4133Controller",
                 name: str,
                 **kwargs: Any) -> None:
        super().__init__(parent, name, **kwargs)
        self._available_planes = ["AB", "BC", "AC"]

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

        self.add_parameter("vector_mode_plane",
                           set_cmd="VM {}",
                           vals=Enum(*self._available_planes),
                           docstring="sets plane of motion for the motors")

        self.add_parameter("vector_position",
                           set_cmd="VP {},{}",  # make group param
                           vals=Ints(-2147483648, 2147483647),
                           units="quadrature counts",
                           docstring="sets position vector for the motion")

        self.add_parameter("vector_acceleration",
                           get_cmd="VA ?",
                           get_parser=int,
                           set_cmd="VA {}",
                           vals=Enum(np.linspace(1024, 1073740800, 1024)),
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
                           vals=Enum(np.linspace(2, 15000000, 2)),
                           units="counts/sec",
                           docstring="sets and gets defined vector's speed")

    @staticmethod
    def _parse_coordinate_system_active(val: str) -> str:
        """
        parses the the current active coordinate system
        """
        if int(val):
            return "T"
        else:
            return "S"

    def vector_seq_end(self) -> None:
        """
        indicates to the controller that the end of the vector is coming up.
        is required to exit the vector mode gracefully
        """
        self.write("VE")


class Motor(InstrumentChannel):
    """
    Class to control motors independently
    """

    def __init__(self,
                 parent: "DMC4133Controller",
                 name: str,
                 **kwargs: Any) -> None:
        super().__init__(parent, name, **kwargs)
        self._axis = name

        self.add_parameter("relative_position",
                           units="quadrature counts",
                           get_cmd=f"MG _PR{self._axis}",
                           get_parser=int,
                           set_cmd=self._set_relative_position,
                           vals=Ints(-2147483648, 2147483647),
                           docstring="sets relative position for the motor's "
                                     "move")

        self.add_parameter("speed",
                           units="counts/sec",
                           get_cmd=f"MG _SP{self._axis}",
                           get_parser=int,
                           set_cmd=self._set_speed,
                           vals=Enum(np.linspace(0, 3000000, 2)),
                           docstring="speed for motor's motion")

        self.add_parameter("acceleration",
                           units="counts/sec2",
                           get_cmd=f"MG _AC{self._axis}",
                           get_parser=int,
                           set_cmd=self._set_acceleration,
                           vals=Enum(np.linspace(1024, 1073740800, 1024)),
                           docstring="acceleration for motor's motion")

        self.add_parameter("deceleration",
                           units="counts/sec2",
                           get_cmd=f"MG _DC{self._axis}",
                           get_parser=int,
                           set_cmd=self._set_deceleration,
                           vals=Enum(np.linspace(1024, 1073740800, 1024)),
                           docstring="deceleration for motor's motion")

    def _set_deceleration(self, val: str) -> None:
        """
        set deceleration for the motor's motion
        """
        self.write(f"DC{self._axis}={val}")

    def _set_acceleration(self, val: str) -> None:
        """
        set acceleration for the motor's motion
        """
        self.write(f"AC{self._axis}={val}")

    def _set_speed(self, val: str) -> None:
        """
        sets speed for motor's motion
        """
        self.write(f"SP{self._axis}={val}")

    def _set_relative_position(self, val: str) -> None:
        """
        sets relative position
        """
        self.write(f"PR{self._axis}={val}")

    def off(self) -> None:
        """
        turns motor off
        """
        self.write(f"MO {self._axis}")

    def on_off_status(self) -> str:
        """
        tells motor on off status
        """
        val = self.ask(f"MG _MO{self._axis}")
        if int(val):
            return "off"
        else:
            return "on"

    def servo_here(self) -> None:
        """
        servo at the motor
        """
        self.write(f"SH {self._axis}")


class DMC4133Controller(GalilMotionController):
    """
    Driver for Galil DMC-4133 Controller
    """

    def __init__(self,
                 name: str,
                 address: str,
                 **kwargs: Any) -> None:
        super().__init__(name=name, address=address, **kwargs)

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
                           vals=Enum(np.linspace(2, 2147483646, 2)),
                           docstring="controller will wait for the amount of "
                                     "time specified before executing the next "
                                     "command")

        self._set_default_update_time()
        self.add_submodule("motor_a", Motor(self, "A"))
        self.add_submodule("motor_b", Motor(self, "B"))
        self.add_submodule("motor_c", Motor(self, "C"))
        self.add_submodule("vector_mode", VectorMode(self, "vector_mode"))

        self.connect_message()

    def _set_default_update_time(self) -> None:
        """
        sets sampling period to default value of 1000. sampling period affects
        the AC, AS, AT, DC, FA, FV, HV, JG, KP, NB, NF, NZ, PL, SD, SP, VA,
        VD, VS, WT commands.
        """
        self.write("TM 1000")

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

    def motors_off(self) -> None:
        """
        turn all motors off
        """
        self.write("MO")

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

    def __init__(self,
                 driver: DMC4133Controller,
                 chip: Dict[str, Union[int, float]]) -> None:

        for key in ["length", "width", "rows", "num_terminals_in_row",
                   "terminal_length", "terminal_width",
                   "inter_terminal_distance_for_adjacent_rows"]:
            if key not in list(chip.keys()):
                raise RuntimeError(f"Chip {key} data not present in the chip "
                                   f"dictionary. Chip dictionary should have "
                                   f"following entries: length, width, rows, "
                                   f"num_terminals_in_row, terminal_length, "
                                   f"terminal_width, "
                                   f"inter_terminal_distance_for_adjacent_rows")

        self._chip_length: float = chip["length"]
        self._chip_width: float = chip["width"]
        self._rows: int = chip["rows"]
        self._num_terminals_in_row: int = chip["num_terminals_in_row"]
        self._terminal_length: float = chip["terminal_length"]
        self._terminal_width: float = chip["terminal_width"]
        self._inter_terminal_distance_for_adjacent_rows: float = chip[
            "inter_terminal_distance_for_adjacent_rows"
        ]

        self.driver = driver

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
