from numpy.random import rand
import pytest

from qcodes.instrument.parameter import ParameterWithSetpoints, Parameter
import qcodes.utils.validators as vals


def test_validation_shapes():
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


def test_validation_inconsistent_shape():
    """
    Parameters with shapes inconsistent with their setpoints should not
    validate
    """
    n_points_1 = Parameter('n_points_1', set_cmd=None, vals=vals.Ints())
    n_points_2 = Parameter('n_points_2', set_cmd=None, vals=vals.Ints())

    n_points_1.set(10)
    n_points_2.set(20)

    setpoints_1 = Parameter('setpoints_1', get_cmd=lambda: rand(n_points_1()),
                            vals=vals.Arrays(shape=(n_points_1,)))

    param_with_diff_lenght = ParameterWithSetpoints('param_1',
                                                    get_cmd=lambda:
                                                    rand(n_points_2()),
                                                    setpoints=(setpoints_1,),
                                                    vals=vals.Arrays(
                                                        shape=(n_points_2,)))

    # inconsistent shapes
    expected_err_msg = (r'Shape of output is not consistent '
                        r'with setpoints. Output is shape '
                        r'\(20,\) and setpoints are shape '
                        r'\(10,\)')
    with pytest.raises(ValueError, match=expected_err_msg):
        param_with_diff_lenght.validate_consistent_shape()
    with pytest.raises(ValueError, match=expected_err_msg):
        param_with_diff_lenght.validate(param_with_diff_lenght.get())


def test_validation_wrong_validator():
    """
    If the validator does not match the actual content the validation should
    fail
    """
    n_points_1 = Parameter('n_points_1', set_cmd=None, vals=vals.Ints())
    n_points_2 = Parameter('n_points_2', set_cmd=None, vals=vals.Ints())

    n_points_1.set(10)
    n_points_2.set(20)
    setpoints_1 = Parameter('setpoints_1', get_cmd=lambda: rand(n_points_1()),
                            vals=vals.Arrays(shape=(n_points_1,)))
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


def test_validation_no_validator():
    """
    If a parameter does not use array validators it cannot be validated.
    """
    n_points_1 = Parameter('n_points_1', set_cmd=None, vals=vals.Ints())

    n_points_1.set(10)
    setpoints_1 = Parameter('setpoints_1', get_cmd=lambda: rand(n_points_1()),
                            vals=vals.Arrays(shape=(n_points_1,)))
    # output does not have a validator
    param_without_validator = ParameterWithSetpoints('param_3',
                                                     get_cmd=lambda:
                                                     rand(n_points_1()),
                                                     setpoints=(setpoints_1,))

    expected_err_msg = (r"Can only validate shapes for "
                        r"parameters with Arrays validator. "
                        r"param_3 does not")
    with pytest.raises(ValueError, match=expected_err_msg):
        param_without_validator.validate_consistent_shape()
    # this does not raise as the shape is only validated for arrays
    param_without_validator.validate(param_without_validator.get())


def test_validation_sp_no_validator():
    """
    If the setpoints do not have an Arrays validator validation
    will fail.
    """
    n_points_2 = Parameter('n_points_2', set_cmd=None, vals=vals.Ints())

    n_points_2.set(20)
    # setpoints do not have a validator
    setpoints_2 = Parameter('setpoints_2', get_cmd=lambda: rand(n_points_2()))
    param_sp_without_validator = ParameterWithSetpoints('param_4',
                                                        get_cmd=lambda:
                                                        rand(n_points_2()),
                                                        setpoints=(
                                                            setpoints_2,),
                                                        vals=vals.Arrays(
                                                            shape=(
                                                                n_points_2,)))

    expected_err_msg = (r"Can only validate shapes for "
                        r"parameters with Arrays validator. "
                        r"setpoints_2 is a setpoint")
    with pytest.raises(ValueError, match=expected_err_msg):
        param_sp_without_validator.validate_consistent_shape()
    with pytest.raises(ValueError, match=expected_err_msg):
        param_sp_without_validator.validate(param_sp_without_validator.get())


def test_validation_without_shape():
    """
    If the Arrays validator does not have a shape the validation will fail
    """
    n_points_1 = Parameter('n_points_1', set_cmd=None, vals=vals.Ints())

    n_points_1.set(10)
    setpoints_1 = Parameter('setpoints_1', get_cmd=lambda: rand(n_points_1()),
                            vals=vals.Arrays(shape=(n_points_1,)))
    param_without_shape = ParameterWithSetpoints('param_5',
                                                 get_cmd=lambda:
                                                 rand(n_points_1()),
                                                 setpoints=(setpoints_1,),
                                                 vals=vals.Arrays())
    expected_err_msg = (r"Trying to validate shape but "
                        r"parameter param_5 does not "
                        r"define a shape")
    with pytest.raises(ValueError, match=expected_err_msg):
        param_without_shape.validate_consistent_shape()
    with pytest.raises(ValueError, match=expected_err_msg):
        param_without_shape.validate(param_without_shape.get())


def test_validation_without_sp_shape():
    """
    If the setpoints validator has no shape the validation will fail
    """
    n_points_1 = Parameter('n_points_1', set_cmd=None, vals=vals.Ints())
    n_points_2 = Parameter('n_points_2', set_cmd=None, vals=vals.Ints())

    n_points_1.set(10)
    n_points_2.set(20)

    setpoints_1 = Parameter('setpoints_1', get_cmd=lambda: rand(n_points_1()),
                            vals=vals.Arrays())
    param_sp_without_shape = ParameterWithSetpoints('param_6',
                                                    get_cmd=lambda:
                                                    rand(n_points_1()),
                                                    setpoints=(setpoints_1,),
                                                    vals=vals.Arrays(
                                                        shape=(
                                                            n_points_1,)))
    expected_err_msg = (r"One or more dimensions have unknown shape "
                        r"when comparing output: \(10,\) to setpoints: "
                        r"\(None,\)")
    with pytest.raises(ValueError, match=expected_err_msg):
        param_sp_without_shape.validate_consistent_shape()
    with pytest.raises(ValueError, match=expected_err_msg):
        param_sp_without_shape.validate(param_sp_without_shape.get())


def test_validation_one_dim_missing():
    """
    If one or more dims of the output does not have a shape the validation
    will fail.
    """
    n_points_1 = Parameter('n_points_1', set_cmd=None, vals=vals.Ints())
    n_points_2 = Parameter('n_points_2', set_cmd=None, vals=vals.Ints())

    n_points_1.set(10)
    n_points_2.set(20)
    setpoints_1 = Parameter('setpoints_1', get_cmd=lambda: rand(n_points_1()),
                            vals=vals.Arrays(shape=(n_points_1, n_points_2)))
    param_sp_without_shape = ParameterWithSetpoints('param_6',
                                                    get_cmd=lambda:
                                                    rand(n_points_1()),
                                                    setpoints=(setpoints_1,),
                                                    vals=vals.Arrays(
                                                        shape=(
                                                            n_points_1, None)))
    expected_err_msg = (r"One or more dimensions have unknown shape "
                        r"when comparing output: \(10, None\) to setpoints: "
                        r"\(10, 20\)")
    with pytest.raises(ValueError, match=expected_err_msg):
        param_sp_without_shape.validate_consistent_shape()
    with pytest.raises(ValueError, match=expected_err_msg):
        param_sp_without_shape.validate(param_sp_without_shape.get())


def test_validation_one_sp_dim_missing():
    """
    If one or more setpoint validators has no shape the validation will fail.
    """
    n_points_1 = Parameter('n_points_1', set_cmd=None, vals=vals.Ints())
    n_points_2 = Parameter('n_points_2', set_cmd=None, vals=vals.Ints())

    n_points_1.set(10)
    n_points_2.set(20)
    setpoints_1 = Parameter('setpoints_1', get_cmd=lambda: rand(n_points_1()),
                            vals=vals.Arrays(shape=(n_points_1, None)))
    param_sp_without_shape = ParameterWithSetpoints('param_6',
                                                    get_cmd=lambda:
                                                    rand(n_points_1()),
                                                    setpoints=(setpoints_1,),
                                                    vals=vals.Arrays(
                                                        shape=(
                                                            n_points_1,
                                                            n_points_2)))
    expected_err_msg = (r"One or more dimensions have unknown shape "
                        r"when comparing output: \(10, 20\) to setpoints: "
                        r"\(10, None\)")
    with pytest.raises(ValueError, match=expected_err_msg):
        param_sp_without_shape.validate_consistent_shape()
    with pytest.raises(ValueError, match=expected_err_msg):
        param_sp_without_shape.validate(param_sp_without_shape.get())
