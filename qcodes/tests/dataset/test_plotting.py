import numpy as np
from hypothesis import given, example, assume, settings, HealthCheck
from hypothesis.strategies import text, sampled_from, floats, lists, data, \
    one_of, just

import qcodes as qc
from qcodes.dataset.plotting import _make_rescaled_ticks_and_units, \
    _ENGINEERING_PREFIXES, _UNITS_FOR_RESCALING

from qcodes.dataset.plotting import (plot_by_id, _appropriate_kwargs,
    _complex_to_real_preparser)
from qcodes.dataset.measurements import Measurement
from qcodes.tests.instrument_mocks import DummyInstrument


@given(param_name=text(min_size=1, max_size=10),
       param_label=text(min_size=0, max_size=15),
       scale=sampled_from(sorted(list(_ENGINEERING_PREFIXES.keys()))),
       unit=sampled_from(sorted(list(
           _UNITS_FOR_RESCALING.union(
               ['', 'unit', 'kg', '%', 'permille', 'nW'])))),
       data_strategy=data()
       )
@example(param_name='huge_param',
         param_label='Larger than the highest scale',
         scale=max(list(_ENGINEERING_PREFIXES.keys())),
         unit='V',
         data_strategy=np.random.random((5,))
                       * 10 ** (3 + max(list(_ENGINEERING_PREFIXES.keys()))))
@example(param_name='small_param',
         param_label='Lower than the lowest scale',
         scale=min(list(_ENGINEERING_PREFIXES.keys())),
         unit='V',
         data_strategy=np.random.random((5,))
                       * 10 ** (-3 + min(list(_ENGINEERING_PREFIXES.keys()))))
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_rescaled_ticks_and_units(scale, unit,
                                  param_name, param_label, data_strategy):
    if isinstance(data_strategy, np.ndarray):
        # No need to generate data, because it is being passed
        data_array = data_strategy
    else:
        # Generate actual data
        scale_upper_bound = 999.9999 * 10 ** scale
        scale_lower_bound = 10 ** scale
        data_array = np.array(data_strategy.draw(
                    lists(elements=one_of(
                        floats(max_value=scale_upper_bound,
                               min_value=-scale_upper_bound),
                        just(np.nan)
                    ), min_size=1)))

        data_array_not_nans = data_array[np.logical_not(np.isnan(data_array))]

        # data_array with nan values *only* is not supported
        assume(len(data_array_not_nans) > 0)

        # data_array has to contain at least one number within the scale of
        # interest (meaning absolute value of at least one number has to
        # be larger than the lowest value within the scale of interest)
        assume((scale_lower_bound < np.abs(data_array_not_nans)).any())

    data_dict = {
        'name': param_name,
        'label': param_label,
        'unit': unit,
        'data': data_array
    }

    ticks_formatter, label = _make_rescaled_ticks_and_units(data_dict)
    if unit in _UNITS_FOR_RESCALING:
        expected_prefix = _ENGINEERING_PREFIXES[scale]
    else:
         if scale != 0:
             expected_prefix = f'$10^{{{scale:.0f}}}$ '
         else:
             expected_prefix = ''
    if param_label == '':
        base_label = param_name
    else:
        base_label = param_label
    postfix = expected_prefix + unit
    if postfix != '':
        assert f"{base_label} ({postfix})" == label
    else:
        assert f"{base_label}" == label

    assert '5' == ticks_formatter(5 / (10 ** (-scale)))
    assert '1' == ticks_formatter(1 / (10 ** (-scale)))
    # also test the fact that "{:g}" is used in ticks formatter function
    assert '2.12346' == ticks_formatter(2.123456789 / (10 ** (-scale)))


def test_plot_by_id_line_and_heatmap(experiment, request):
    """
    Test that line plots and heatmaps can be plotted together
    """
    inst = DummyInstrument('dummy', gates=['s1', 'm1', 's2', 'm2'])
    request.addfinalizer(inst.close)

    inst.m1.get = np.random.randn
    inst.m2.get = lambda: np.random.randint(0, 5)

    meas = Measurement()
    meas.register_parameter(inst.s1)
    meas.register_parameter(inst.s2)
    meas.register_parameter(inst.m2, setpoints=(inst.s1, inst.s2) )
    meas.register_parameter(inst.m1, setpoints=(inst.s1,))

    with meas.run() as datasaver:
        for outer in range(10):
            datasaver.add_result((inst.s1, outer),
                                (inst.m1, inst.m1()))
            for inner in range(10):
                datasaver.add_result((inst.s1, outer),
                                    (inst.s2, inner),
                                    (inst.m2, inst.m2()))

    dataid = datasaver.run_id
    plot_by_id(dataid)
    plot_by_id(dataid, cmap='bone')


def test_appropriate_kwargs():

    kwargs = {'cmap': 'bone'}
    check = kwargs.copy()

    with _appropriate_kwargs('1D_line', False, **kwargs) as ap_kwargs:
        assert ap_kwargs == {}

    assert kwargs == check

    with _appropriate_kwargs('1D_point', False, **kwargs) as ap_kwargs:
        assert ap_kwargs == {}

    assert kwargs == check

    with _appropriate_kwargs('1D_bar', False, **kwargs) as ap_kwargs:
        assert ap_kwargs == {}

    assert kwargs == check

    with _appropriate_kwargs('2D_grid', False, **kwargs) as ap_kwargs:
        assert ap_kwargs == kwargs

    assert kwargs == check

    with _appropriate_kwargs('2D_point', False, **{}) as ap_kwargs:
        assert len(ap_kwargs) == 1
        assert ap_kwargs['cmap'] == qc.config.plotting.default_color_map


def test__complex_to_real_preparser_complex_toplevel_param():

    data_in = [[{'data': np.array([0, 1, 2]),
                 'name': 'voltage',
                 'label': 'swept voltage',
                 'unit': 'V'},
                {'data': np.array([0+0j, 1+2j, -1+1j]),
                 'name': 'signal',
                 'label': 'complex signal',
                 'unit': 'Ohm'}]]

    data_out = _complex_to_real_preparser(data_in, conversion='real_and_imag')

    assert np.shape(data_out) == (2, 2)
    assert data_out[0][0] == data_in[0][0]

    real_param = data_out[0][1]
    assert real_param['name'] == 'signal_real'
    assert real_param['label'] =='complex signal [real]'
    assert all(real_param['data'] == np.array([0, 1, -1]))
    assert real_param['unit'] == 'Ohm'

    imag_param = data_out[1][1]
    assert imag_param['name'] == 'signal_imag'
    assert imag_param['label'] == 'complex signal [imag]'
    assert all(imag_param['data'] == np.array([0, 2, 1]))
    assert imag_param['unit'] == 'Ohm'

    data_out = _complex_to_real_preparser(data_in, conversion='mag_and_phase')

    assert len(data_out) == 2
    assert len(data_out[0]) == 2
    assert data_out[0][0] == data_in[0][0]

    phase_param = data_out[1][1]
    assert phase_param['name'] == 'signal_phase'
    assert phase_param['label'] =='complex signal [phase]'
    assert all(phase_param['data'] == np.angle(np.array([0+0j, 1+2j, -1+1j])))
    assert phase_param['unit'] == 'rad'

    mag_param = data_out[0][1]
    assert mag_param['name'] == 'signal_mag'
    assert mag_param['label'] == 'complex signal [mag]'
    assert all(mag_param['data'] == np.array([0, np.sqrt(5), np.sqrt(2)]))
    assert mag_param['unit'] == 'Ohm'

    data_out = _complex_to_real_preparser(data_in, conversion='mag_and_phase',
                                          degrees=True)

    phase_param = data_out[1][1]
    assert phase_param['name'] == 'signal_phase'
    assert phase_param['label'] =='complex signal [phase]'
    assert all(phase_param['data'] == np.angle(np.array([0+0j, 1+2j, -1+1j]),
                                               deg=True))
    assert phase_param['unit'] == 'deg'


def test__complex_to_real_preparser_complex_setpoint():

    data_in = [[{'data': np.array([0+0j, 1+2j, -1+1j]),
                 'name': 'signal',
                 'label': 'complex signal',
                 'unit': 'Ohm'},
                 {'data': np.array([0, 1, 2]),
                 'name': 'voltage',
                 'label': 'measured voltage',
                 'unit': 'V'}
                ]]

    data_out = _complex_to_real_preparser(data_in, conversion='real_and_imag')

    assert np.shape(data_out) == (1, 3)
    assert data_out[0][-1] == data_in[0][-1]

    real_param = data_out[0][0]
    assert real_param['name'] == 'signal_real'
    assert real_param['label'] =='complex signal [real]'
    assert all(real_param['data'] == np.array([0, 1, -1]))
    assert real_param['unit'] == 'Ohm'

    imag_param = data_out[0][1]
    assert imag_param['name'] == 'signal_imag'
    assert imag_param['label'] == 'complex signal [imag]'
    assert all(imag_param['data'] == np.array([0, 2, 1]))
    assert imag_param['unit'] == 'Ohm'

    measured_param = data_out[0][2]
    assert measured_param['name'] == 'voltage'
    assert measured_param['label'] == 'measured voltage'
    assert measured_param['unit'] == 'V'
    assert all(measured_param['data'] == np.array([0, 1, 2]))
