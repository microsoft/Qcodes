from numpy.random import rand
import pytest


from qcodes.instrument.parameter import ParameterWithSetpoints, Parameter
import qcodes.utils.validators as vals





def test_verification_shapes():
    """
    Test that various parameters with setpoints and invalid shape combinations
    raises when validated.
    """

    n_points_1 = Parameter('n_points_1', set_cmd=None, vals=vals.Ints())
    n_points_2 = Parameter('n_points_2', set_cmd=None, vals=vals.Ints())

    n_points_1.set(10)
    n_points_2.set(20)

    setpoints_1 = Parameter('setpoints_1', get_cmd=lambda: rand(n_points_1()),
                          vals=vals.Arrays(shape=(n_points_1,)))
    setpoints_2 = Parameter('setpoints_2', get_cmd=lambda: rand(n_points_2()),
                          vals=vals.Arrays(shape=(n_points_2,)))

    param_with_setpoints_1 = ParameterWithSetpoints('param_1',
                                                  get_cmd=lambda:
                                                  rand(n_points_1()),
                                                  setpoints=(setpoints_1,),
                                                  vals=vals.Arrays(shape=(n_points_1,)))

    # the two shapes are the same so validation works
    param_with_setpoints_1.validate_consistent_shape()
    param_with_setpoints_1.validate(param_with_setpoints_1.get())

    param_with_setpoints_2 = ParameterWithSetpoints('param_2',
                                                    get_cmd=lambda:
                                                    rand(n_points_1(),
                                                         n_points_2()),
                                                    setpoints=(setpoints_1,
                                                               setpoints_2),
                                                    vals=vals.Arrays(
                                                    shape=(n_points_1,
                                                           n_points_2)))
    # 2d
    param_with_setpoints_2.validate_consistent_shape()
    param_with_setpoints_2.validate(param_with_setpoints_2.get())

    param_with_setpoints_3 = ParameterWithSetpoints('param_3',
                                                    get_cmd=lambda:
                                                    rand(n_points_2()),
                                                    setpoints=(setpoints_1,),
                                                    vals=vals.Arrays(
                                                        shape=(n_points_2,)))

    # inconsistent shapes
    with pytest.raises(ValueError):
        param_with_setpoints_3.validate_consistent_shape()
    with pytest.raises(ValueError):
        param_with_setpoints_3.validate(param_with_setpoints_3.get())

    # output is not consistent with validator
    param_with_setpoints_4 = ParameterWithSetpoints('param_4',
                                                    get_cmd=lambda:
                                                    rand(n_points_2()),
                                                    setpoints=(setpoints_1,),
                                                    vals=vals.Arrays(
                                                        shape=(n_points_1,)))

    # this does not raise because the validator shapes are consistent
    param_with_setpoints_4.validate_consistent_shape()
    # but the output is not consistent with the validator
    with pytest.raises(ValueError):
        param_with_setpoints_4.validate(param_with_setpoints_4())