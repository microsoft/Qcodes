from numpy.random import rand
import pytest


from qcodes.instrument.parameter import ParameterWithSetpoints, Parameter
import qcodes.utils.validators as vals





def test_verification_shapes():
    """
    Test that various parameters with setpoints and shape combinations
    validate correctly.
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
                                                    vals=vals.Arrays(
                                                        shape=(n_points_1,)))

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


def test_verification_inconsistent_shape():

    n_points_1 = Parameter('n_points_1', set_cmd=None, vals=vals.Ints())
    n_points_2 = Parameter('n_points_2', set_cmd=None, vals=vals.Ints())

    n_points_1.set(10)
    n_points_2.set(20)

    setpoints_1 = Parameter('setpoints_1', get_cmd=lambda: rand(n_points_1()),
                            vals=vals.Arrays(shape=(n_points_1,)))
    setpoints_2 = Parameter('setpoints_2', get_cmd=lambda: rand(n_points_2()))

    param_with_diff_lenght = ParameterWithSetpoints('param_1',
                                                    get_cmd=lambda:
                                                    rand(n_points_2()),
                                                    setpoints=(setpoints_1,),
                                                    vals=vals.Arrays(
                                                        shape=(n_points_2,)))

    # inconsistent shapes
    with pytest.raises(ValueError, match=r'Shape of output is not consistent '
                                         r'with setpoints\. Output is shape '
                                         r'\(20,\) and setpoints are shape '
                                         r'\(10,\)'):
        param_with_diff_lenght.validate_consistent_shape()
    with pytest.raises(ValueError, match=r'Shape of output is not consistent '
                                         r'with setpoints\. Output is shape '
                                         r'\(20,\) and setpoints are shape '
                                         r'\(10,\)'):
        param_with_diff_lenght.validate(param_with_diff_lenght.get())

    # output is not consistent with validator
    param_with_wrong_validator = ParameterWithSetpoints('param_2',
                                                        get_cmd=lambda:
                                                        rand(n_points_2()),
                                                        setpoints=(
                                                            setpoints_1,),
                                                        vals=vals.Arrays(
                                                            shape=(
                                                                n_points_1,)))

    # this does not raise because the validator shapes are consistent
    param_with_wrong_validator.validate_consistent_shape()
    # but the output is not consistent with the validator
    with pytest.raises(ValueError, match=r'does not have expected shape'
                                         r' \(10,\), '
                                         r'it has shape \(20,\); '
                                         r'Parameter: param_2'):
        param_with_wrong_validator.validate(param_with_wrong_validator())

    # output does not have a validator
    param_without_validator = ParameterWithSetpoints('param_3',
                                                     get_cmd=lambda:
                                                     rand(n_points_1()),
                                                     setpoints=(setpoints_1,))

    with pytest.raises(ValueError, match=r"Can only validate shapes for "
                                         r"parameters with Arrays validator. "
                                         r"param_3 does not"):
        param_without_validator.validate_consistent_shape()

    # setpoints do not have a validator
    param_sp_without_validator = ParameterWithSetpoints('param_4',
                                                        get_cmd=lambda:
                                                        rand(n_points_2()),
                                                        setpoints=(
                                                            setpoints_2,),
                                                        vals=vals.Arrays(
                                                            shape=(
                                                                n_points_1,)))

    with pytest.raises(ValueError, match=r"Can only validate shapes for "
                                         r"parameters with Arrays validator. "
                                         r"setpoints_2 is a setpoint"):
        param_sp_without_validator.validate_consistent_shape()
