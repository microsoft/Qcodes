import numpy as np
from numpy.random import rand
import pytest

from qcodes.instrument.parameter import ParameterWithSetpoints, Parameter,\
    expand_setpoints_helper
import qcodes.utils.validators as vals


@pytest.fixture()
def parameters():
    n_points_1 = Parameter('n_points_1', set_cmd=None, vals=vals.Ints())
    n_points_2 = Parameter('n_points_2', set_cmd=None, vals=vals.Ints())
    n_points_3 = Parameter('n_points_3', set_cmd=None, vals=vals.Ints())

    n_points_1.set(10)
    n_points_2.set(20)
    n_points_3.set(15)

    setpoints_1 = Parameter('setpoints_1', get_cmd=lambda: np.arange(n_points_1()),
                            vals=vals.Arrays(shape=(n_points_1,)))
    setpoints_2 = Parameter('setpoints_2', get_cmd=lambda: np.arange(n_points_2()),
                            vals=vals.Arrays(shape=(n_points_2,)))
    setpoints_3 = Parameter('setpoints_3', get_cmd=lambda: np.arange(n_points_3()),
                            vals=vals.Arrays(shape=(n_points_3,)))
    yield (n_points_1, n_points_2, n_points_3,
           setpoints_1, setpoints_2, setpoints_3)



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
    assert "<Arrays, shape: (<qcodes.instrument.parameter." \
           "Parameter: n_points_1 at" in param_with_setpoints_1.__doc__

    # the two shapes are the same so validation works
    param_with_setpoints_1.validate_consistent_shape()
    param_with_setpoints_1.validate(param_with_setpoints_1.get())

    param_with_setpoints_2 = ParameterWithSetpoints('param_2',
                                                    get_cmd=lambda:
                                                    rand(n_points_1(),
                                                         n_points_2()),
                                                    vals=vals.Arrays(
                                                        shape=(n_points_1,
                                                               n_points_2)))

    param_with_setpoints_2.setpoints = (setpoints_1, setpoints_2)
    # 2d
    param_with_setpoints_2.validate_consistent_shape()
    param_with_setpoints_2.validate(param_with_setpoints_2.get())


def test_setpoints_non_parameter_raises():

    """
    Test that putting some random function as a setpoint parameter will
    raise as expected.
    """

    n_points_1 = Parameter('n_points_1', set_cmd=None, vals=vals.Ints())
    n_points_2 = Parameter('n_points_2', set_cmd=None, vals=vals.Ints())

    n_points_1.set(10)
    n_points_2.set(20)


    err_msg = (r"Setpoints is of type <class 'function'> "
               r"expcected a QCoDeS parameter")
    with pytest.raises(TypeError, match=err_msg):
        param_with_setpoints_1 = ParameterWithSetpoints('param_1',
                                                        get_cmd=lambda:
                                                        rand(n_points_1()),
                                                        setpoints=(
                                                        lambda x: x,),
                                                        vals=vals.Arrays(
                                                            shape=(
                                                                n_points_1,)))

    param_with_setpoints_1 = ParameterWithSetpoints('param_1',
                                                    get_cmd=lambda:
                                                    rand(n_points_1()),
                                                    vals=vals.Arrays(
                                                        shape=(
                                                            n_points_1,)))

    with pytest.raises(TypeError, match=err_msg):
        param_with_setpoints_1.setpoints = (lambda x: x,)



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
                        r'\(<qcodes.instrument.parameter.Parameter: n_points_2 at [0-9]+>,\) '
                        r'and setpoints are shape '
                        r'\(<qcodes.instrument.parameter.Parameter: n_points_1 at [0-9]+>,\)')
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

    with pytest.raises(ValueError, match=r"A ParameterWithSetpoints must have "
                                         r"an Arrays validator got "
                                         r"<class 'NoneType'>"):
        param_without_validator = ParameterWithSetpoints('param_3',
                                                         get_cmd=lambda:
                                                         rand(n_points_1()),
                                                         setpoints=(
                                                         setpoints_1,))



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
    with pytest.raises(RuntimeError, match=r"A ParameterWithSetpoints must "
                                           r"have a shape defined "
                                           r"for its validator."):
        param_without_shape = ParameterWithSetpoints('param_5',
                                                     get_cmd=lambda:
                                                     rand(n_points_1()),
                                                     setpoints=(setpoints_1,),
                                                     vals=vals.Arrays())


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
                        r"when comparing output: \(<qcodes.instrument.parameter"
                        r".Parameter: n_points_1 at [0-9]+>,\) to setpoints: "
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
                        r"when comparing output: \(<qcodes.instrument.parameter.Parameter: n_points_1 at [0-9]+>, None\) to setpoints: "
                        r"\(<qcodes.instrument.parameter.Parameter: n_points_1 at [0-9]+>, <qcodes.instrument.parameter.Parameter: n_points_2 at [0-9]+>\)")
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
                        r"when comparing output: \(<qcodes.instrument.parameter.Parameter: n_points_1 at [0-9]+>, <qcodes.instrument.parameter.Parameter: n_points_2 at [0-9]+>\) to setpoints: "
                        r"\(<qcodes.instrument.parameter.Parameter: n_points_1 at [0-9]+>, None\)")
    with pytest.raises(ValueError, match=expected_err_msg):
        param_sp_without_shape.validate_consistent_shape()
    with pytest.raises(ValueError, match=expected_err_msg):
        param_sp_without_shape.validate(param_sp_without_shape.get())


def test_expand_setpoints_1c(parameters):
    """
    Test that the setpoints expander helper function works correctly
    """

    n_points_1, n_points_2, n_points_3, \
    setpoints_1, setpoints_2, setpoints_3 = parameters

    param_with_setpoints_1 = ParameterWithSetpoints('param_1',
                                                    get_cmd=lambda:
                                                    rand(n_points_1()),
                                                    setpoints=(setpoints_1,),
                                                    vals=vals.Arrays(
                                                        shape=(n_points_1,)))

    data = expand_setpoints_helper(param_with_setpoints_1)

    assert len(data) == 2
    assert len(data[0][1]) == len(data[1][1])


def test_expand_setpoints_2d(parameters):

    n_points_1, n_points_2, n_points_3, \
    setpoints_1, setpoints_2, setpoints_3 = parameters

    param_with_setpoints_2 = ParameterWithSetpoints('param_2',
                                                    get_cmd=lambda:
                                                    rand(n_points_1(),
                                                         n_points_2()),
                                                    vals=vals.Arrays(
                                                        shape=(n_points_1,
                                                               n_points_2)))
    param_with_setpoints_2.setpoints = (setpoints_1, setpoints_2)

    data = expand_setpoints_helper(param_with_setpoints_2)

    assert len(data) == 3
    assert data[0][1].shape == data[1][1].shape
    assert data[0][1].shape == data[2][1].shape

    sp1 = data[0][1]
    sp2 = data[1][1]
    # the first set of setpoints should be repeated along the second axis
    for i in range(sp1.shape[1]):
        np.testing.assert_array_equal(sp1[:, i], np.arange(sp1.shape[0]))
    # the second set of setpoints should be repeated along the first axis
    for i in range(sp2.shape[0]):
        np.testing.assert_array_equal(sp2[i, :], np.arange(sp1.shape[1]))


def test_expand_setpoints_3d(parameters):

    n_points_1, n_points_2, n_points_3, \
        setpoints_1, setpoints_2, setpoints_3 = parameters

    param_with_setpoints_3 = ParameterWithSetpoints('param_2',
                                                    get_cmd=lambda:
                                                    rand(n_points_1(),
                                                         n_points_2(),
                                                         n_points_3()),
                                                    vals=vals.Arrays(
                                                        shape=(n_points_1,
                                                               n_points_2,
                                                               n_points_3)))
    param_with_setpoints_3.setpoints = (setpoints_1, setpoints_2, setpoints_3)
    data = expand_setpoints_helper(param_with_setpoints_3)
    assert len(data) == 4
    assert data[0][1].shape == data[1][1].shape
    assert data[0][1].shape == data[2][1].shape
    assert data[0][1].shape == data[3][1].shape

    sp1 = data[0][1]
    for i in range(sp1.shape[1]):
        for j in range(sp1.shape[2]):
            np.testing.assert_array_equal(sp1[:, i, j], np.arange(sp1.shape[0]))
    sp2 = data[1][1]
    for i in range(sp2.shape[0]):
        for j in range(sp2.shape[2]):
            np.testing.assert_array_equal(sp2[i, :, j], np.arange(sp2.shape[1]))

    sp3 = data[2][1]
    for i in range(sp3.shape[0]):
        for j in range(sp3.shape[1]):
            np.testing.assert_array_equal(sp3[i, j, :], np.arange(sp3.shape[2]))
