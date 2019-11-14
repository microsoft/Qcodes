"""
These are the basic black box tests for the doNd functions.
"""

from qdev_wrappers.dataset.doNd import do0d, do1d, do2d
from qcodes.instrument.parameter import Parameter
from qcodes import config, new_experiment
from qcodes.utils import validators

import pytest

config.user.mainfolder = "output"  # set ouput folder for doNd's
new_experiment("doNd-tests", sample_name="no sample")

_param = Parameter('simple_parameter',
                   set_cmd=None,
                   get_cmd=lambda: 1)

_paramComplex = Parameter('simple_complex_parameter',
                   set_cmd=None,
                   get_cmd=lambda: 1 + 1j,
                   vals=validators.ComplexNumbers())

_param_set = Parameter('simple_setter_paramater',
                       set_cmd=None,
                       get_cmd=None)


def _param_func():
    _new_param = Parameter('modified_parameter',
                           set_cmd= None,
                           get_cmd= lambda: _param.get()*2)
    assert _new_param.get() == 2
    return _new_param


def test_do0d():

    do0d(_param)
    do0d(_param, write_period=1)
    do0d(_param, write_period=1, do_plot=False)

    do0d(_paramComplex)
    do0d(_paramComplex, write_period=1)
    do0d(_paramComplex, write_period=1, do_plot=False)

    do0d(_param_func(), write_period=1, do_plot=False)

    do0d(_param, _paramComplex, write_period=1, do_plot=False)
    do0d(_param_func(), _paramComplex, write_period=1, do_plot=False)


def test_do1d():

    _start = 0
    _stop = 1
    _num_points = 1
    _delay_list = [0, 0.1, 1]

    for _delay in _delay_list:
        do1d(_param_set, _start, _stop, _num_points, _delay, _param)
        do1d(_param_set, _start, _stop, _num_points, _delay, _paramComplex)
        do1d(_param_set, _start, _stop, _num_points, _delay, _param,
                                                                 _paramComplex)


