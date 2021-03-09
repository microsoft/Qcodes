"""
This file holds the QCoDeS driver for the Galil DMC-41x3 motor controllers,
colloquially known as the "stepper motors".
"""
from typing import Any, Dict, Optional, List
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel
from qcodes.instrument.group_parameter import GroupParameter, Group
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
        Open connection to Galil motion controller. This method assumes that
        the initial mapping of Galil motion controller's hardware's mapping
        to an IP address is done using GDK and the IP address in burned in.
        This applies that Motion controller no more requests for an IP address
        and a connection to the Motion controller can be done by the IP
        address burned in.
        """
        self.g.GOpen(self.address + ' --direct -s ALL')

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

    def timeout(self, val: int) -> None:
        """
        Sets timeout for the instrument

        Args:
            val: time in milliseconds
        """
        if val < 1:
            raise RuntimeError("Timeout can not be less than 1 ms")

        self.g.GTimeout(val)

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
                           get_cmd=None,
                           set_cmd="CS {}",
                           vals=Enum("S", "T"),
                           docstring="clears vectors specified in the given "
                                     "coordinate system")

        self.add_parameter("vector_mode_plane",
                           get_cmd=None,
                           set_cmd="VM {}",
                           vals=Enum(*self._available_planes),
                           docstring="sets plane of motion for the motors")

        self.add_parameter("vec_pos_first_coordinate",
                           unit="quadrature counts",
                           vals=Ints(-2147483648, 2147483647),
                           parameter_class=GroupParameter,
                           docstring="sets vector position for plane's first"
                                     "axis. e.g., if vector_mode_plane "
                                     "is specified 'AC'. this param sets "
                                     "vector position for 'A' axis to be used"
                                     "in motion")

        self.add_parameter("vec_pos_second_coordinate",
                           unit="quadrature counts",
                           vals=Ints(-2147483648, 2147483647),
                           parameter_class=GroupParameter,
                           docstring="sets vector position for plane's second"
                                     "axis. e.g., if vector_mode_plane "
                                     "is specified 'AC'. this param sets "
                                     "vector position for 'C' axis to be used"
                                     "in motion")

        self._vector_position = Group([self.vec_pos_first_coordinate,
                                       self.vec_pos_second_coordinate],
                                      set_cmd="VP {vec_pos_first_coordinate},"
                                              "{vec_pos_second_coordinate}",
                                      get_cmd=None)

        self.add_parameter("vector_acceleration",
                           get_cmd="VA ?",
                           get_parser=int,
                           set_cmd="VA {}",
                           vals=Enum(np.linspace(1024, 1073740800, 1024)),
                           unit="counts/sec2",
                           docstring="sets and gets the defined vector's "
                                     "acceleration")

        self.add_parameter("vector_deceleration",
                           get_cmd="VD ?",
                           get_parser=int,
                           set_cmd="VD {}",
                           vals=Enum(np.linspace(1024, 1073740800, 1024)),
                           unit="counts/sec2",
                           docstring="sets and gets the defined vector's "
                                     "deceleration")

        self.add_parameter("vector_speed",
                           get_cmd="VS ?",
                           get_parser=int,
                           set_cmd="VS {}",
                           vals=Enum(np.linspace(2, 15000000, 2)),
                           unit="counts/sec",
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

    def begin_seq(self) -> None:
        """
        begins motion of the motor
        """
        self.write("BG S")

    def after_seq_motion(self) -> None:
        """
        wait till motion ends
        """
        self.write("AM S")


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
                           unit="quadrature counts",
                           get_cmd=f"MG _PR{self._axis}",
                           get_parser=int,
                           set_cmd=self._set_relative_position,
                           vals=Ints(-2147483648, 2147483647),
                           docstring="sets relative position for the motor's "
                                     "move")

        self.add_parameter("speed",
                           unit="counts/sec",
                           get_cmd=f"MG _SP{self._axis}",
                           get_parser=int,
                           set_cmd=self._set_speed,
                           vals=Enum(np.linspace(0, 3000000, 2)),
                           docstring="speed for motor's motion")

        self.add_parameter("acceleration",
                           unit="counts/sec2",
                           get_cmd=f"MG _AC{self._axis}",
                           get_parser=int,
                           set_cmd=self._set_acceleration,
                           vals=Enum(np.linspace(1024, 1073740800, 1024)),
                           docstring="acceleration for motor's motion")

        self.add_parameter("deceleration",
                           unit="counts/sec2",
                           get_cmd=f"MG _DC{self._axis}",
                           get_parser=int,
                           set_cmd=self._set_deceleration,
                           vals=Enum(np.linspace(1024, 1073740800, 1024)),
                           docstring="deceleration for motor's motion")

        self.add_parameter("homing_velocity",
                           unit="counts/sec",
                           get_cmd=f"MG _HV{self._axis}",
                           get_parser=int,
                           set_cmd=self._set_homing_velocity,
                           vals=Enum(np.linspace(0, 3000000, 2)),
                           docstring="sets the slew speed for the FI "
                                     "final move to the index and all but the "
                                     "first stage of HM (home)")

    def _set_homing_velocity(self, val: str) -> None:
        """
        sets the slew speed for the FI final move to the index and all but
        the first stage of HM.
        """
        self.write(f"HV{self._axis}={val}")

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

    def _setup_spm(self) -> None:
        """
        sets up for Stepper Position Maintenance (SPM) mode
        """
        # Set the profiler to stop axis upon error
        self.write(f"OE{self._axis}=1")
        self.write(f"KS{self._axis}=16")  # Set step smoothing
        self.write(f"MT{self._axis}=-2")  # Motor type set to stepper
        self.write(f"YA{self._axis}=1")   # Step resolution of the drive

        # Motor resolution (full steps per revolution)
        self.write(f"YB{self._axis}=200")
        # Encoder resolution (counts per revolution)
        self.write(f"YC{self._axis}=4000")

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

    def begin(self) -> None:
        """
        begins motion of the motor
        """
        self.write(f"BG {self._axis}")

    def after_motion(self) -> None:
        """
        wait till motion ends
        """
        self.write(f"AM {self._axis}")

    def home(self) -> None:
        """
        performs a three stage homing sequence for servo systems and a two
        stage sequence for stepper motor.

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
        self.speed(2000)
        self.homing_velocity(256)

        # home command
        self.write(f"HM {self._axis}")

        # begin motion
        self.begin()

        # wait for motion to finish
        self.after_motion()

        # end the program
        self.root_instrument.end_program()

    def enable_stepper_position_maintenance_mode(self, motor: str) -> None:
        """
        enables Stepper Position Maintenance mode and allows for error
        correction when error happens
        """
        self._setup_spm()
        self.servo_here()  # Enable axis
        self.root_instrument.wait(50)  # Allow slight settle time
        self.write(f"YS{self._axis}=1")

    def error_magnitude(self) -> float:
        """
        gives the magnitude of error, in drive step counts, for axes in
        Stepper Position Maintenance mode.

        a step count is directly proportional to the micro-stepping
        resolution of the stepper drive.
        """
        return float(self.ask(f"QS{self._axis}=?"))


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
                           get_cmd=None,
                           set_cmd="PF 10.{}",
                           vals=Ints(0, 4),
                           docstring="sets number of decimals in the format "
                                     "of the position")

        self.add_parameter("absolute_position",
                           get_cmd=self._get_absolute_position,
                           set_cmd=None,
                           unit="quadrature counts",
                           docstring="gets absolute position of the motors "
                                     "from the set origin")

        self.add_parameter("wait",
                           get_cmd=None,
                           set_cmd="WT {}",
                           unit="ms",
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

    def begin_motors(self) -> None:
        """
        begin motion of all motors simultaneously
        """
        self.write("BG")


class ArmHeadPosition:

    def __init__(self, x: int, y: int, z: int, *args) -> None:
        self._x = x
        self._y = y
        self._z = z

    def get_x(self) -> int:
        return self._x

    def set_x(self, val: int) -> None:
        self._x = val

    def get_y(self) -> int:
        return self._y

    def set_y(self, val: int) -> None:
        self._y = val

    def get_z(self) -> int:
        return self._z

    def set_z(self, val: int) -> None:
        self._z = val


class Arm:

    def __init__(self,
                 controller: DMC4133Controller,
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

        self.controller = controller
        self._arm_head_status = None

    def _pick_arm_head_up(self) -> None:
        """
        picks up arm head if down
        """
        if self._arm_head_status == "up":
            self.log.info("Arm head is already up.")
            return

        # implement logic to move in up direction

        self._arm_head_status = "up"

    def _move_arm_head(self) -> None:
        """
        move from current position to next position
        """
        x1, y1, z1 = (self._current_position.get_x(),
                      self._current_position.get_y(),
                      self._current_position.get_z())

        x2, y2, z2 = (self._next_position.get_x(),
                      self._next_position.get_y(),
                      self._next_position.get_z())

        if x1 == x2 and y1 == y2 and z1 == z2:
            self.log.info("Arm head already at the required location.")
            return

        assert self._arm_head_status == "up"

        # implement logic to move

    def _put_arm_head_down(self) -> None:
        """
        puts arm head down if up
        """
        if self._arm_head_status == "down":
            self.log.info("Arm head is already down.")
            return

        # implement logic to move in down direction

        self._arm_head_status = "down"

    def move_to_next_row(self) -> None:
        """
        moves motors to next row of pads
        """
        self._pick_arm_head_up()
        self._move_arm_head()
        self._put_arm_head_down()

    def set_begin_position(self) -> None:
        """
        sets first row of pads in chip as begin position
        """
        self.controller.define_position_as_origin()
        self._begin_position = (0, 0, 0)
        self._current_position = ArmHeadPosition(0, 0, 0)
        self._arm_head_status = "down"

    def set_end_position(self) -> None:
        """
        sets last row of pads in chip as end position
        """
        pos_dict = self.controller.absolute_position()
        self._end_position = (pos_dict["A"], pos_dict["B"], pos_dict["C"])
        self._current_position.set_x(pos_dict["A"])
        self._current_position.set_y(pos_dict["B"])
        self._current_position.set_z(pos_dict["C"])
        self._arm_head_status = "down"
