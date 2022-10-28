---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# QCoDeS example with Galil DMC4133 Controller

+++

Purpose of this notebook is to demonstrate how Galil DMC4133 Controller driver along with Arm class can be used for running measurements while controlling the arm head simulteneously. Before begining,

1. Make sure that you have `gclib` package installed in your environment. If not, then follow the instructions [here](https://www.galil.com/sw/pub/all/doc/gclib/html/python.html) for installation.
2. Make sure that the controller is connected to your PC through an Ethernet cable and the configuration is set as according to these instructions on Windows operating system.
    
        a. Go to Control Panel -> Network and Internet -> Network Connections and select the appropriate network adapter.
        b. Next go to the Properties of that adapter, and then Properties for Internet Protocol Version 4 (TCP/IPv4).
        c. Select "Use the following IP address" and add an IP address and Subnet. (If the Galil has an IP address of 10.10.10.100 burned in, you would need a PC IP address of something like 10.10.10.1 with a subnet of 255.255.255.0.)

Once, the connection to the DMC4133 Controller is established, we can begin with necessary imports and the calibration process. So, let us begin!

+++

# Imports

```{code-cell} ipython3
from qcodes.instrument_drivers.Galil.dmc_41x3 import DMC4133Controller, Arm
```

# DMC4133 Controller

```{code-cell} ipython3
controller = DMC4133Controller(name='controller', address='192.168.8.74')
```

The controller with 3 motors can be initialized as above. Now lets discuss the features available on the QCoDeS DMC4133 Controller driver.

We can find out the absolute position of the controller with following call.

```{code-cell} ipython3
controller.absolute_position()
```

Current position of the controller can ber defined as origin with command:

```{code-cell} ipython3
controller.define_position_as_origin()
```

`tell_error` method on the controller can be used to get which error occured in case it does in a human readable form.

```{code-cell} ipython3
controller.tell_error()
```

`stop` method can be used to stop motion of all motors simultaneously. On this call, the motors will decelerate to a stop. 

```{code-cell} ipython3
controller.stop()
```

`abort` call can be used to abort any motion on all three motors simultaneously. On this call, the motors will immediately come to a halt.

```{code-cell} ipython3
controller.abort()
```

`motors_off` call turns all motors off simultaneously.

```{code-cell} ipython3
controller.motors_off()
```

`wait_till_motion_complete` method call is a blocking call and it does what it says - waits till motion on all motors complete.

```{code-cell} ipython3
controller.wait_till_motion_complete()
```

## Motor submodules on the Controller

+++

There are 3 motor submodules on the DMC4133 controller driver each corresponding to a motor attached to the controller. They are named `motor_a`, `motor_b` and `motor_c`. All the submodules are identical except for the motor axis they control. Can be accessed as follows:

```{code-cell} ipython3
A = controller.motor_a
B = controller.motor_b
C = controller.motor_c
```

Let us take the example of motor A and see what all features are there on each submodule.

While setting up the motion on the motor, `relative_position` parameter will help set the position to the motor where the motion should end as relative to the current position.

```{code-cell} ipython3
A.relative_position(2000)  #values are in quadrature counts
```

Set relative position can be found out by the following call.

```{code-cell} ipython3
A.relative_position()
```

Motor's motion `speed`, `acceleration` and `deceleration` can be set with parameters as follows.

```{code-cell} ipython3
A.speed(2000)     #value in quadrature counts per sec and should be a multiple of 2
A.acceleration(2048)  #value in quadrature counts per sec sq. and should be a multiple of 1024
A.deceleration(2048)  # same as acceleration
```

Now, the motion parameters are set. Before commanding the motor to move we need to switch on them using the following method call.

```{code-cell} ipython3
A.servo_here()
```

After this motor can be commanded to move with..

```{code-cell} ipython3
A.begin()
```

We can check if a motor is in motion with following command.

```{code-cell} ipython3
A.is_in_motion() # returns 1 if in motion otherwise 0
```

A blocking wait command can be given which waits till the motor's motion completes.

```{code-cell} ipython3
A.wait_till_motor_motion_complete()
```

To switch off the motor, call..

```{code-cell} ipython3
A.off()
```

In order to find out if the motor is in on or off state, we can use following method.

```{code-cell} ipython3
A.on_off_status()
```

For each motor, reverse and forward software limits can be set with the help of `reverse_sw_limit` and `forward_sw_limit` parameters respectively. The values should be given in quadrature counts. The motor will not move beyond these set limits.

+++

Each motor can be set to turn off when an error occurs with the following parameter.

```{code-cell} ipython3
A.off_when_error_occurs("disable")  # possible arguments are: "disable",
                                    #                         "enable for position, amp error or abort"
                                    #                         "enable for hw limit switch"
                                    #                         "enable for all" 
```

Error magnitude can be checked on each motor with following method.

```{code-cell} ipython3
A.error_magnitude()
```

## Vector mode submodules on the Controller

+++

On DMC4133 Controller there are three vector mode submodules present for planer movement in `AB`, `BC` and `AC` planes. They can be accessed as follows.

```{code-cell} ipython3
AB = controller.plane_ab
BC = controller.plane_bc
AC = controller.plane_ac
```

Let us consider `BC` plane to see what all features are available on the driver. Before we setup the motion, that plane needs to be activated with following method call.

```{code-cell} ipython3
BC.activate()
```

There are two possible coordinate systems `S` and `T` which can be set for the movement.

```{code-cell} ipython3
BC.coordinate_system('S')
```

Now, let us set up the motion on this plane.

```{code-cell} ipython3
# all units are in quadrature counts

BC.vector_position(2000,3000)    # first coordinate corresponds to B axis and second to C axis in this case
BC.vector_speed(2000)
BC.vector_acceleration(2048)
BC.vector_deceleration(2048)
BC.vector_seq_end()              # this call is necessary to exit the vector mode gracefully
```

Since the motion is setup, we can instruct the motors to move with following call.

```{code-cell} ipython3
BC.begin_seq()
```

To clear the sequence of commands from a given coordinate system, use:

```{code-cell} ipython3
BC.clear_sequence('S')
```

Our tour of the `DMC4133Controller` class ends here. Lets move on to `Arm` Class. But before moving forward, keep in mind the following assumptions:

    1. Needle arm head is assumed to be rectangular in shape with one or more rows and each row with one or more needles.
    2. Chip to be probed is assumed to be rectangular as well with number of rows in it to be multiple of number of rows in the needle head and number of pads in each row to be a multiple of number of needles in each row of the needle head.  

+++

# Arm class

```{code-cell} ipython3
arm = Arm(controller)
```

Now that we have imported an initialized the controller and the arm. We need to calibrate the arm.

+++

# Calibration

+++

Check the state of the motors.

```{code-cell} ipython3
print(controller.absolute_position())
print(controller.motor_a.on_off_status())
print(controller.motor_b.on_off_status())
print(controller.motor_c.on_off_status())
```

```{code-cell} ipython3
controller.motors_off()
```

```{code-cell} ipython3
arm.set_arm_kinematics()  # sets default values of arm speed to be 100 micro meters per second,
                          # acceleration and deceleration to be 2048 micro meters per second square
```

Manually move the motors and take the needle head to the position where you want to set the origin. This will be the left bottom corner of the chip and run following commands.

```{code-cell} ipython3
controller.define_position_as_origin()
```

```{code-cell} ipython3
arm.set_left_bottom_position()
```

From now on all the motor movements will be controlled by the driver commands.

Next step is to set reverse limits for for all three motors. First take the motor C to the extreme reverse position which you want to set as the reverse limit with following command.

```{code-cell} ipython3
arm.move_motor_c_by(distance=-1000)    # distance is in micro meters
```

Now that you are at the position which you want to set as the reverse limit for motor C, run the following command.

```{code-cell} ipython3
arm.set_motor_c_reverse_limit()
```

For setting forward limit following command can be run.

```{code-cell} ipython3
arm.set_motor_c_forward_limit()
```

Repeat the same process for motor A but you need to set both forward and reverse limits at the desired locations.

```{code-cell} ipython3
arm.move_motor_a_by(distance=-200)
```

```{code-cell} ipython3
arm.set_motor_a_forward_limit()
```

```{code-cell} ipython3
arm.set_motor_a_reverse_limit()
```

Now, for motor B. Again both forward and reverse limits need to be set at desired locations.

```{code-cell} ipython3
arm.move_motor_b_by(distance=-2000)
```

```{code-cell} ipython3
arm.set_motor_b_forward_limit()
```

```{code-cell} ipython3
arm.set_motor_b_reverse_limit()
```

You have set the reverse limits for all three motors. Next we will define the chip plane. We have already set the chip left bottom corner as the origin of the system. Now, we will set the left top corner first and then right top corner.

Move individual motors with following commands.

```{code-cell} ipython3
arm.move_motor_a_by(distance=-1000)
```

```{code-cell} ipython3
arm.move_motor_b_by(distance=3000)
```

```{code-cell} ipython3
arm.move_motor_c_by(distance=-300)
```

When you are satisfied that the motor is at the left top position of the chip. Run the following command.

```{code-cell} ipython3
arm.set_left_top_position()
```

Again, move invidual motors with the above mentioned commands and when you are satisfied that the arm needle is at the right top position of the chip, run the following command.

```{code-cell} ipython3
arm.set_right_top_position()
```

You have not set the boundaries for the motion of the motor. Though the calibration process is not complete yet. You need to set the chip details.

+++

## Set chip details

```{code-cell} ipython3
arm.rows = 2
arm.pads = 3
arm.inter_row_distance = arm.norm_b            # since there are only 2 rows
arm.inter_pad_distance = arm.norm_c / 2        # since there are 3 pads per row
```

Calibration is complete! Remember you are at the right top position of the chip crrently. Move the needle head to left bottom position and before that set pick up diatance.

```{code-cell} ipython3
arm.set_pick_up_distance()
```

```{code-cell} ipython3
arm.move_towards_left_bottom_position()
```

# Integration with measurement process

```{code-cell} ipython3
import qcodes as qc
from qcodes import (
    Measurement,
    initialise_or_create_database_at,
    load_or_create_experiment,
)
from qcodes.tests.instrument_mocks import DummyInstrument, DummyInstrumentWithMeasurement
```

```{code-cell} ipython3
station = qc.Station()
```

```{code-cell} ipython3
# A dummy instrument dac with two parameters ch1 and ch2
dac = DummyInstrument('dac', gates=['ch1', 'ch2'])

# A dummy instrument that generates some real looking output depending
# on the values set on the setter_instr, in this case the dac
dmm = DummyInstrumentWithMeasurement('dmm', setter_instr=dac)
```

```{code-cell} ipython3
station.add_component(dac)
station.add_component(dmm)
```

```{code-cell} ipython3
initialise_or_create_database_at("~/experiments.db")
```

```{code-cell} ipython3
exp = load_or_create_experiment(experiment_name='galil_controller_testing',
                                sample_name="no sample")
```

Our arm is set up and the measurement is set up. Run the following block for measurement as we are at the 1st row.

```{code-cell} ipython3
meas = Measurement(exp=exp, station=station, name='xyz_measurement')
meas.register_parameter(dac.ch1)  # register the first independent parameter
meas.register_parameter(dmm.v1, setpoints=(dac.ch1,))  # now register the dependent oone

meas.write_period = 2

with meas.run() as datasaver:
    for set_v in np.linspace(0, 25, 10):
        dac.ch1.set(set_v)
        get_v = dmm.v1.get()
        datasaver.add_result((dac.ch1, set_v),
                             (dmm.v1, get_v))

    dataset = datasaver.dataset  # convenient to have for plotting
```

Now you have option to move to the next row or pad with following commands.

```{code-cell} ipython3
arm.move_to_row(1)
```

```{code-cell} ipython3
arm.move_to_pad(3)
```

Once this motion is complete, you can use individual motors commands for minor adjustments. 

```{code-cell} ipython3
arm.move_motor_a_by(distance=15)
```

```{code-cell} ipython3
arm.move_motor_b_by(distance=50)
```

```{code-cell} ipython3
arm.move_motor_c_by(distance=-5)
```

Repeat the measurement code block for the new row or pad.

+++

When you are done with the all the measurements, use following command to close the controller.

```{code-cell} ipython3
arm.controller.close()
```
