"""
This file holds the QCoDeS driver for the Galil DMC-41x3 motor controllers.

Colloquially known as the "stepper motors".
"""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from qcodes.instrument import Instrument, InstrumentBaseKWArgs, InstrumentChannel
from qcodes.validators import Enum, Ints, Multiples

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.parameters import Parameter

try:
    import gclib  # pyright: ignore[reportMissingImports]
except ImportError as e:
    raise ImportError(
        "Cannot find gclib library. Download gclib installer from "
        "https://www.galil.com/sw/pub/all/rn/gclib.html and install Galil "
        "motion controller software for your OS. Afterwards go "
        "to https://www.galil.com/sw/pub/all/doc/gclib/html/python.html and "
        "follow instruction to be able to import gclib package in your "
        "environment."
    ) from e


class GalilMotionController(Instrument):
    """
    Base class for Galil Motion Controller drivers
    """

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[InstrumentBaseKWArgs]"
    ) -> None:
        """
        Initializes and opens the connection to the Galil Motion Controller

        Args:
            name: name for the instrument
            address: address of the controller
            **kwargs: kwargs are forwarded to base class.
        """
        super().__init__(name=name, **kwargs)
        self.g = gclib.py()
        self._address = address
        self.open()
        self.connect_message()

    def open(self) -> None:
        """
        Open connection to Galil motion controller. This method assumes that
        the initial mapping of Galil motion controller's hardware's
        to an IP address is done using GDK and the IP address is burned in.
        This applies that Motion controller no more requests for an IP address
        and a connection to the Motion controller can be done by the IP
        address burned in.
        """
        self.g.GOpen(self._address + " --direct -s ALL")

    def get_idn(self) -> dict[str, str | None]:
        """
        Get Galil motion controller hardware information
        """
        data = self.g.GInfo().split(" ")
        idparts: list[str | None] = [
            "Galil Motion Control, Inc.",
            data[1],
            data[4],
            data[3][:-1],
        ]

        return dict(zip(("vendor", "model", "serial", "firmware"), idparts))

    def write_raw(self, cmd: str) -> None:
        """
        Write for Galil motion controller
        """
        self.g.GCommand(cmd + "\r")

    def ask_raw(self, cmd: str) -> str:
        """
        Asks/Reads data from Galil motion controller
        """
        return self.g.GCommand(cmd + "\r")

    def timeout(self, val: int) -> None:
        """
        Sets timeout for the instrument

        Args:
            val: time in milliseconds. -1 disables the timeout
        """
        if val < -1:
            raise RuntimeError("Timeout value cannot be set less than -1")

        self.g.GTimeout(val)

    def close(self) -> None:
        """
        Close connection to the instrument
        """
        self.g.GClose()

    def motion_complete(self, axes: str) -> None:
        """
        Wait for motion to complete for given axes, e.g. ``"A"``
        for axis A, ``"ABC"`` for axes A, B, and C
        """
        self.g.GMotionComplete(axes)


class GalilDMC4133VectorMode(InstrumentChannel):
    """
    Class to control motors in vector mode
    """

    def __init__(
        self,
        parent: "GalilDMC4133Controller",
        name: str,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        """
        Initializes the vector mode submodule for the controller

        Args:
            parent: an instance of DMC4133Controller
            name: name of the vector mode plane
            **kwargs: kwargs are forwarded to base class.
        """
        super().__init__(parent, name, **kwargs)
        self._plane = name
        self._vector_position_validator = Ints(
            min_value=-2147483648, max_value=2147483647
        )

        self.coordinate_system: Parameter = self.add_parameter(
            "coordinate_system",
            get_cmd="CA ?",
            get_parser=self._parse_coordinate_system_active,
            set_cmd="CA {}",
            vals=Enum("S", "T"),
            docstring="activates coordinate system for the motion. Two "
            " coordinate systems are possible with values "
            "'S' and 'T'. All vector mode commands will apply to "
            "the active coordinate system.",
        )
        """activates coordinate system for the motion. Two  coordinate systems are possible with values 'S' and 'T'. All vector mode commands will apply to the active coordinate system."""

        self.vector_acceleration: Parameter = self.add_parameter(
            "vector_acceleration",
            get_cmd="VA ?",
            get_parser=lambda s: int(float(s)),
            set_cmd="VA {}",
            vals=Multiples(min_value=1024, max_value=1073740800, divisor=1024),
            unit="counts/sec2",
            docstring="sets and gets the defined vector's acceleration",
        )
        """sets and gets the defined vector's acceleration"""

        self.vector_deceleration: Parameter = self.add_parameter(
            "vector_deceleration",
            get_cmd="VD ?",
            get_parser=lambda s: int(float(s)),
            set_cmd="VD {}",
            vals=Multiples(min_value=1024, max_value=1073740800, divisor=1024),
            unit="counts/sec2",
            docstring="sets and gets the defined vector's deceleration",
        )
        """sets and gets the defined vector's deceleration"""

        self.vector_speed: Parameter = self.add_parameter(
            "vector_speed",
            get_cmd="VS ?",
            get_parser=lambda s: int(float(s)),
            set_cmd="VS {}",
            vals=Multiples(min_value=2, max_value=15000000, divisor=2),
            unit="counts/sec",
            docstring="sets and gets defined vector's speed",
        )
        """sets and gets defined vector's speed"""

    @staticmethod
    def _parse_coordinate_system_active(val: str) -> str:
        """
        parses the current active coordinate system
        """
        if int(float(val)):
            return "T"
        else:
            return "S"

    def activate(self) -> None:
        """
        activate plane of motion
        """
        self.write(f"VM {self._plane}")

    def vector_position(self, first_coord: int, second_coord: int) -> None:
        """
        sets the final vector position for the motion considering current
        position as the origin
        """
        self._vector_position_validator.validate(first_coord)
        self._vector_position_validator.validate(second_coord)

        self.write(f"VP {first_coord},{second_coord}")

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

    def clear_sequence(self, coord_sys: str) -> None:
        """
        clears vectors specified in the given coordinate system
        """
        if coord_sys not in ["S", "T"]:
            raise RuntimeError(
                f"possible coordinate systems are 'S' or 'T'. "
                f"you provided the following value instead: "
                f" '{coord_sys}'"
            )

        self.write(f"CS {coord_sys}")


VectorMode = GalilDMC4133VectorMode
"""
Alias for backwards compatibility
"""


class GalilDMC4133Motor(InstrumentChannel):
    """
    Class to control a single motor (independent of possible other motors)
    """

    def __init__(
        self,
        parent: "GalilDMC4133Controller",
        name: str,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        """
        Initializes individual motor submodules

        Args:
            parent: an instance of DMC4133Controller
            name: name of the motor to be controlled
            **kwargs: kwargs are forwarded to base class.
        """
        super().__init__(parent, name, **kwargs)
        self._axis = name

        self.relative_position: Parameter = self.add_parameter(
            "relative_position",
            unit="quadrature counts",
            get_cmd=f"MG _PR{self._axis}",
            get_parser=lambda s: int(float(s)),
            set_cmd=self._set_relative_position,
            vals=Ints(-2147483648, 2147483647),
            docstring="sets relative position for the motor's move",
        )
        """sets relative position for the motor's move"""

        self.speed: Parameter = self.add_parameter(
            "speed",
            unit="counts/sec",
            get_cmd=f"MG _SP{self._axis}",
            get_parser=lambda s: int(float(s)),
            set_cmd=self._set_speed,
            vals=Multiples(min_value=0, max_value=3000000, divisor=2),
            docstring="speed for motor's motion",
        )
        """speed for motor's motion"""

        self.acceleration: Parameter = self.add_parameter(
            "acceleration",
            unit="counts/sec2",
            get_cmd=f"MG _AC{self._axis}",
            get_parser=lambda s: int(float(s)),
            set_cmd=self._set_acceleration,
            vals=Multiples(min_value=1024, max_value=1073740800, divisor=1024),
            docstring="acceleration for motor's motion",
        )
        """acceleration for motor's motion"""

        self.deceleration: Parameter = self.add_parameter(
            "deceleration",
            unit="counts/sec2",
            get_cmd=f"MG _DC{self._axis}",
            get_parser=lambda s: int(float(s)),
            set_cmd=self._set_deceleration,
            vals=Multiples(min_value=1024, max_value=1073740800, divisor=1024),
            docstring="deceleration for motor's motion",
        )
        """deceleration for motor's motion"""

        self.off_when_error_occurs: Parameter = self.add_parameter(
            "off_when_error_occurs",
            get_cmd=self._get_off_when_error_occurs,
            set_cmd=self._set_off_when_error_occurs,
            val_mapping={
                "disable": 0,
                "enable for position, amp error or abort": 1,
                "enable for hw limit switch": 2,
                "enable for all": 3,
            },
            docstring="enables or disables the motor to "
            "automatically turn off when error occurs",
        )
        """enables or disables the motor to automatically turn off when error occurs"""

        self.reverse_sw_limit: Parameter = self.add_parameter(
            "reverse_sw_limit",
            unit="quadrature counts",
            get_cmd=f"MG _BL{self._axis}",
            get_parser=lambda s: int(float(s)),
            set_cmd=self._set_reverse_sw_limit,
            vals=Ints(-2147483648, 2147483647),
            docstring="can be used to set software reverse limit for the motor."
            " motor motion will stop beyond this limit automatically."
            " default value is -2147483648. this value effectively "
            "disables the reverse limit.",
        )
        """can be used to set software reverse limit for the motor. motor motion will stop beyond this limit automatically. default value is -2147483648. this value effectively disables the reverse limit."""

        self.forward_sw_limit: Parameter = self.add_parameter(
            "forward_sw_limit",
            unit="quadrature counts",
            get_cmd=f"MG _FL{self._axis}",
            get_parser=lambda s: int(float(s)),
            set_cmd=self._set_forward_sw_limit,
            vals=Ints(-2147483648, 2147483647),
            docstring="can be used to set software forward limit for the motor."
            " motor motion will stop beyond this limit automatically."
            " default value is 2147483647. this value effectively "
            "disables the forward limit.",
        )
        """can be used to set software forward limit for the motor. motor motion will stop beyond this limit automatically. default value is 2147483647. this value effectively disables the forward limit."""

    def _set_reverse_sw_limit(self, val: int) -> None:
        """
        Sets reverse software limit
        """
        self.write(f"BL{self._axis}={val}")

    def _set_forward_sw_limit(self, val: int) -> None:
        """
        Sets forward software limit
        """
        self.write(f"FL{self._axis}={val}")

    def _get_off_when_error_occurs(self) -> int:
        """
        Gets the status if motor is automatically set to turn off when error
        occurs.
        """
        val = self.ask(f"MG _OE{self._axis}")

        return int(val[0])

    def _set_off_when_error_occurs(self, val: int) -> None:
        """
        sets the motor to turn off automatically when the error occurs
        """
        self.write(f"OE{self._axis}={val}")

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
        if val[0] == "1":
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
        begins motion of the motor (returns immediately)
        """
        self.write(f"BG {self._axis}")

    def is_in_motion(self) -> int:
        """
        checks if the motor is in motion or not. return 1, if motor is in
        motion otherwise 0
        """
        return int(float(self.ask(f"MG _BG{self._axis}")))

    def wait_till_motor_motion_complete(self) -> None:
        """
        wait for motion on the motor to complete
        """
        try:
            while self.is_in_motion():
                pass
        except KeyboardInterrupt:
            self.root_instrument.abort()
            self.off()

    def error_magnitude(self) -> float:
        """
        gives the magnitude of error, in quadrature counts, for this motor
        a count is directly proportional to the micro-stepping
        resolution of the stepper drive.
        """
        return float(self.ask(f"QS{self._axis}=?"))


Motor = GalilDMC4133Motor
"""
Alias for backwards compatibility
"""


class GalilDMC4133Controller(GalilMotionController):
    """
    Driver for Galil DMC-4133 Controller
    """

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[InstrumentBaseKWArgs]"
    ) -> None:
        """
        Initializes the DMC4133Controller class

        Args:
            name: name for the instance
            address: address of the controller burned in
            **kwargs: kwargs are forwarded to base class.
        """
        super().__init__(name=name, address=address, **kwargs)

        self.position_format_decimals: Parameter = self.add_parameter(
            "position_format_decimals",
            get_cmd=None,
            set_cmd="PF 10.{}",
            vals=Ints(0, 4),
            docstring="sets number of decimals in the format of the position",
        )
        """sets number of decimals in the format of the position"""

        self.absolute_position: Parameter = self.add_parameter(
            "absolute_position",
            get_cmd=self._get_absolute_position,
            set_cmd=None,
            unit="quadrature counts",
            docstring="gets absolute position of the motors from the set origin",
        )
        """gets absolute position of the motors from the set origin"""

        self.wait: Parameter = self.add_parameter(
            "wait",
            get_cmd=None,
            set_cmd="WT {}",
            unit="ms",
            vals=Multiples(min_value=2, max_value=2147483646, divisor=2),
            docstring="controller will wait for the amount of "
            "time specified before executing the next "
            "command",
        )
        """controller will wait for the amount of time specified before executing the next command"""

        self._set_default_update_time()
        self.add_submodule("motor_a", GalilDMC4133Motor(self, "A"))
        self.add_submodule("motor_b", GalilDMC4133Motor(self, "B"))
        self.add_submodule("motor_c", GalilDMC4133Motor(self, "C"))
        self.add_submodule("plane_ab", GalilDMC4133VectorMode(self, "AB"))
        self.add_submodule("plane_bc", GalilDMC4133VectorMode(self, "BC"))
        self.add_submodule("plane_ac", GalilDMC4133VectorMode(self, "AC"))

    def _set_default_update_time(self) -> None:
        """
        sets sampling period to default value of 1000. sampling period affects
        the AC, AS, AT, DC, FA, FV, HV, JG, KP, NB, NF, NZ, PL, SD, SP, VA,
        VD, VS, WT commands.
        """
        self.write("TM 1000")

    def _get_absolute_position(self) -> dict[str, int]:
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

    def wait_till_motion_complete(self) -> None:
        """
        this method waits for the motion on all motors to complete
        """
        try:
            while (
                int(float(self.ask("MG _BGA")))
                or int(float(self.ask("MG _BGB")))
                or int(float(self.ask("MG _BGC")))
            ):
                pass
        except KeyboardInterrupt:
            self.abort()
            self.motors_off()


DMC4133Controller = GalilDMC4133Controller
"""
Alias for backwards compatibility
"""


class GalilDMC4133Arm:
    """
    Module to control probe arm. It is assumed that the chip to be probed has
    one or more rows with each row having one or more pads. Design of the
    chip to be probed should resemble the following::

        |--------------------------|
        |--------------------------|
        |--------------------------|
        |--------------------------|
        |--------------------------|

    Needle head for the arm in assumed to have rows of needles which is a
    divisor of the number of rows in the chip and number of needles in each
    row of the needle head is a divisor for the number of pads in each row
    of the chip.
    """

    def __init__(self, controller: GalilDMC4133Controller) -> None:
        """
        Initializes the arm class

        Args:
            controller: an instance of DMC4133Controller
        """
        self.controller = controller

        # initialization (all these points will have values in quadrature
        # counts)
        self._left_bottom_position: tuple[int, int, int] | None = None
        self._left_top_position: tuple[int, int, int] | None = None
        self._right_top_position: tuple[int, int, int] | None = None

        # motion directions (all these values are in quadrature counts)
        self._a: np.ndarray  # right_top - left_bottom
        self._b: np.ndarray  # left_top - left_bottom
        self._c: np.ndarray  # right_top - left_top
        self._n: np.ndarray
        self.norm_a: float
        self.norm_b: float
        self.norm_c: float

        # eqn of the chip plane (in quadrature counts)
        self._plane_eqn: np.ndarray

        # current vars
        self._current_row: int | None = None
        self._current_pad: int | None = None

        # chip details
        self.rows: int
        self.pads: int
        self.inter_row_distance: float  # in quadrature counts
        self.inter_pad_distance: float  # in quadrature counts

        # arm kinematics (in the units of quadrature counts)
        self._speed: int
        self._acceleration: int
        self._deceleration: int

        self._arm_pick_up_distance: int

        self._target: np.ndarray

    @property
    def current_row(self) -> int | None:
        return self._current_row

    @property
    def current_pad(self) -> int | None:
        return self._current_pad

    @property
    def left_bottom_position(self) -> tuple[int, int, int] | None:
        return self._left_bottom_position

    @property
    def left_top_position(self) -> tuple[int, int, int] | None:
        return self._left_top_position

    @property
    def right_top_position(self) -> tuple[int, int, int] | None:
        return self._right_top_position

    @property
    def arm_pick_up_distance(self) -> int:
        return self._arm_pick_up_distance

    @property
    def speed(self) -> int:
        return self._speed

    @property
    def acceleration(self) -> int:
        return self._acceleration

    @property
    def deceleration(self) -> int:
        return self._deceleration

    def set_arm_kinematics(
        self, speed: int = 100, acceleration: int = 2048, deceleration: int = 2048
    ) -> None:
        """
        sets the arm kinematics values for speed, acceleration and
        deceleration in micro meters/sec and micro meters/sec^2 respectively
        """

        if acceleration % 256 != 0 or deceleration % 256 != 0:
            raise RuntimeError(
                "Acceleration and deceleration must be a multiple of 256. "
                "Units are micro meters/sec^2"
            )

        self._speed = _convert_micro_meter_to_quadrature_counts(speed)
        self._acceleration = _convert_micro_meter_to_quadrature_counts(acceleration)
        self._deceleration = _convert_micro_meter_to_quadrature_counts(deceleration)

    def set_pick_up_distance(self, distance: float = 3000) -> None:
        """
        sets pick up distance in micrometers for the arm
        """

        self._arm_pick_up_distance = _convert_micro_meter_to_quadrature_counts(distance)

    def set_left_bottom_position(self) -> None:
        pos = self.controller.absolute_position()
        self._left_bottom_position = (pos["A"], pos["B"], pos["C"])

        self._calculate_ortho_vector()

    def set_left_top_position(self) -> None:
        pos = self.controller.absolute_position()
        self._left_top_position = (pos["A"], pos["B"], pos["C"])

        self._calculate_ortho_vector()

    def set_right_top_position(self) -> None:
        pos = self.controller.absolute_position()
        self._right_top_position = (pos["A"], pos["B"], pos["C"])

        self._calculate_ortho_vector()

    def _calculate_ortho_vector(self) -> None:
        if (
            self._left_bottom_position is None
            or self._left_top_position is None
            or self._right_top_position is None
        ):
            return

        right_top_position: npt.NDArray[np.int32] = np.asarray(self._right_top_position)
        left_bottom_position: npt.NDArray[np.int32] = np.asarray(
            self._left_bottom_position
        )
        left_top_position: npt.NDArray[np.int32] = np.asarray(self._left_top_position)

        a = right_top_position - left_bottom_position
        self.norm_a = float(np.linalg.norm(a))
        self._a = a / self.norm_a

        b = left_top_position - left_bottom_position
        self.norm_b = float(np.linalg.norm(b))
        self._b = b / self.norm_b

        c = right_top_position - left_top_position
        self.norm_c = float(np.linalg.norm(c))
        self._c = c / self.norm_c

        n = np.cross(self._a, self._b)
        norm_n = np.linalg.norm(n)
        self._n = n / norm_n

        intercept = np.array(sum(-1 * self._n * self._left_bottom_position))
        self._plane_eqn = np.append(self._n, intercept)

    def move_motor_a_by(self, distance: float) -> None:
        """
        Moves motor A by distance given in micro meters
        """
        a = self.controller.motor_a

        d = _convert_micro_meter_to_quadrature_counts(distance)

        a.relative_position(d)
        a.speed(self._speed)
        a.acceleration(self._acceleration)
        a.deceleration(self._deceleration)
        a.servo_here()
        a.begin()
        a.wait_till_motor_motion_complete()

    def move_motor_b_by(self, distance: float) -> None:
        """
        Moves motor B by distance given in micro meters
        """
        b = self.controller.motor_b

        d = _convert_micro_meter_to_quadrature_counts(distance)

        b.relative_position(d)
        b.speed(self._speed)
        b.acceleration(self._acceleration)
        b.deceleration(self._deceleration)
        b.servo_here()
        b.begin()
        b.wait_till_motor_motion_complete()

    def move_motor_c_by(self, distance: float) -> None:
        """
        Moves motor B by distance given in micro meters
        """
        c = self.controller.motor_c

        d = _convert_micro_meter_to_quadrature_counts(distance)

        c.relative_position(d)
        c.speed(self._speed)
        c.acceleration(self._acceleration)
        c.deceleration(self._deceleration)
        c.servo_here()
        c.begin()
        c.wait_till_motor_motion_complete()

    def _setup_motion(self, rel_vec: np.ndarray, d: float, speed: float) -> None:
        """
        sets up motion parameters. all arguments have units in quadrature counts
        """

        pos = self.controller.absolute_position()

        a = int(np.floor(rel_vec[0] * d))
        b = int(np.floor(rel_vec[1] * d))
        c = int(np.floor(rel_vec[2] * d))

        self._target = np.array([pos["A"] + a, pos["B"] + b, pos["C"] + c, 1])

        if np.dot(self._plane_eqn, self._target) < 0:
            flag = 1
            for idx, coord in enumerate([a, b, c]):
                temp1 = coord - 1
                temp2 = coord + 1
                if idx == 0:
                    target1 = np.array(
                        [pos["A"] + temp1, pos["B"] + b, pos["C"] + c, 1]
                    )
                    target2 = np.array(
                        [pos["A"] + temp2, pos["B"] + b, pos["C"] + c, 1]
                    )
                elif idx == 1:
                    target1 = np.array(
                        [pos["A"] + a, pos["B"] + temp1, pos["C"] + c, 1]
                    )
                    target2 = np.array(
                        [pos["A"] + a, pos["B"] + temp2, pos["C"] + c, 1]
                    )
                else:
                    target1 = np.array(
                        [pos["A"] + a, pos["B"] + b, pos["C"] + temp1, 1]
                    )
                    target2 = np.array(
                        [pos["A"] + a, pos["B"] + b, pos["C"] + temp2, 1]
                    )

                if np.dot(self._plane_eqn, target1) >= 0:
                    flag = 0
                    if idx == 0:
                        a = temp1
                    elif idx == 1:
                        b = temp1
                    else:
                        c = temp1
                    break

                if np.dot(self._plane_eqn, target2) >= 0:
                    flag = 0
                    if idx == 0:
                        a = temp2
                    elif idx == 1:
                        b = temp2
                    else:
                        c = temp2
                    break

            if flag:
                raise RuntimeError(
                    f"Cannot move to {self._target[:3]}. Target location is "
                    f"below chip plane."
                )

        sp_a = int(np.floor(abs(rel_vec[0]) * speed))
        if sp_a % 2 != 0:
            sp_a += 1

        sp_b = int(np.floor(abs(rel_vec[1]) * speed))
        if sp_b % 2 != 0:
            sp_b += 1

        sp_c = int(np.floor(abs(rel_vec[2]) * speed))
        if sp_c % 2 != 0:
            sp_c += 1

        acc_a = _calculate_vector_component(vec=rel_vec[0], val=self._acceleration)
        acc_b = _calculate_vector_component(vec=rel_vec[1], val=self._acceleration)
        acc_c = _calculate_vector_component(vec=rel_vec[2], val=self._acceleration)

        dec_a = _calculate_vector_component(vec=rel_vec[0], val=self._deceleration)
        dec_b = _calculate_vector_component(vec=rel_vec[1], val=self._deceleration)
        dec_c = _calculate_vector_component(vec=rel_vec[2], val=self._deceleration)

        motor_a = self.controller.motor_a
        motor_b = self.controller.motor_b
        motor_c = self.controller.motor_c

        motor_a.relative_position(a)
        motor_a.speed(sp_a)
        motor_a.acceleration(acc_a)
        motor_a.deceleration(dec_a)
        motor_a.servo_here()

        motor_b.relative_position(b)
        motor_b.speed(sp_b)
        motor_b.acceleration(acc_b)
        motor_b.deceleration(dec_b)
        motor_b.servo_here()

        motor_c.relative_position(c)
        motor_c.speed(sp_c)
        motor_c.acceleration(acc_c)
        motor_c.deceleration(dec_c)
        motor_c.servo_here()

    def _move(self) -> None:
        self.controller.begin_motors()
        self.controller.wait_till_motion_complete()

    def _pick_up(self) -> None:
        self.move_motor_c_by(distance=-20)
        self._setup_motion(
            rel_vec=self._n, d=self._arm_pick_up_distance, speed=self._speed
        )
        self._move()

    def _put_down(self) -> None:
        motion_vec = -1 * self._n

        pos = self.controller.absolute_position()
        x1 = pos["A"]
        y1 = pos["B"]
        z1 = pos["C"]
        current = np.array([x1, y1, z1, 1])

        denominator = np.sqrt(
            self._plane_eqn[0] ** 2 + self._plane_eqn[1] ** 2 + self._plane_eqn[2] ** 2
        )
        d = abs(sum(self._plane_eqn * current)) / denominator

        self._setup_motion(rel_vec=motion_vec, d=d, speed=self._speed)
        self._move()

    def move_towards_left_bottom_position(self) -> None:
        self._pick_up()

        motion_vec = -1 * self._a
        self._setup_motion(rel_vec=motion_vec, d=self.norm_a, speed=self._speed)
        self._move()
        self._put_down()

        self._current_row = 1
        self._current_pad = 1

    def move_to_next_row(self) -> None:
        if self._current_row is None or self._current_pad is None:
            raise RuntimeError("Current position unknown.")

        if self._current_row == self.rows:
            raise RuntimeError("Cannot move further")

        self._pick_up()

        self._setup_motion(
            rel_vec=self._b, d=self.inter_row_distance, speed=self._speed
        )
        self._move()

        self._put_down()

        self._current_row = self._current_row + 1

    def move_to_begin_row_pad_from_end_row_last_pad(self) -> None:
        if self._current_row is None or self._current_pad is None:
            raise RuntimeError("Current position unknown.")

        if self._current_pad == self.pads:
            raise RuntimeError("Cannot move further")

        motion_vec = -1 * self._b * self.norm_b + self._c * self.inter_pad_distance
        norm = float(np.linalg.norm(motion_vec))
        motion_vec_cap = motion_vec / norm

        self._pick_up()

        self._setup_motion(rel_vec=motion_vec_cap, d=norm, speed=self._speed)
        self._move()

        self._put_down()

        self._current_row = 1
        self._current_pad = self._current_pad + 1

    def move_to_row(self, num: int) -> None:
        if num < 1 or num > self.rows:
            raise RuntimeError(
                f"Row num: {num} is out of range. Row numbers start from 1 "
                f"and max number of rows is {self.rows}."
            )

        sign = 1
        assert self._current_row is not None
        d = (num - self._current_row) * self.inter_row_distance

        if d == 0:
            raise RuntimeError(
                f"You are at the row where you want to be. Current row number "
                f"is {self._current_row}."
            )
        if d < 0:
            sign = -1
            d = abs(d)

        self._pick_up()

        self._setup_motion(rel_vec=sign * self._b, d=d, speed=self._speed)
        self._move()

        self._put_down()

        self._current_row = num

    def move_to_pad(self, num: int) -> None:
        if num < 1 or num > self.pads:
            raise RuntimeError(
                f"Pad num: {num} is out of range. Pad number start from 1 "
                f"and max number is {self.pads}."
            )

        sign = 1
        assert self._current_pad is not None
        d = (num - self._current_pad) * self.inter_pad_distance

        if d == 0:
            raise RuntimeError(
                f"You are at the pad where you want to be. Current pad "
                f"number is {self._current_pad}."
            )
        if d < 0:
            sign = -1
            d = abs(d)

        self._pick_up()

        self._setup_motion(rel_vec=sign * self._c, d=d, speed=self._speed)
        self._move()

        self._put_down()

        self._current_pad = num

    def set_motor_a_forward_limit(self) -> None:
        pos = self.controller.absolute_position()
        self.controller.motor_a.forward_sw_limit(pos["A"])

    def set_motor_a_reverse_limit(self) -> None:
        pos = self.controller.absolute_position()
        self.controller.motor_a.reverse_sw_limit(pos["A"])

    def set_motor_b_forward_limit(self) -> None:
        pos = self.controller.absolute_position()
        self.controller.motor_b.forward_sw_limit(pos["B"])

    def set_motor_b_reverse_limit(self) -> None:
        pos = self.controller.absolute_position()
        self.controller.motor_b.reverse_sw_limit(pos["B"])

    def set_motor_c_forward_limit(self) -> None:
        pos = self.controller.absolute_position()
        self.controller.motor_c.forward_sw_limit(pos["C"])

    def set_motor_c_reverse_limit(self) -> None:
        pos = self.controller.absolute_position()
        self.controller.motor_c.reverse_sw_limit(pos["C"])


def _convert_micro_meter_to_quadrature_counts(val: float) -> int:
    return int(20 * val)


def _calculate_vector_component(vec: float, val: int) -> int:
    return_val = int(np.floor(abs(vec) * val))
    return_val = return_val + 1024
    rem = return_val % 1024
    return_val = return_val - rem

    assert return_val % 1024 == 0

    return return_val


Arm = GalilDMC4133Arm
"""
Alias for backwards compatibility
"""
