import numpy as np

from qcodes.dataset.plotting import _make_rescaled_ticks_and_units, \
    _ENGINEERING_PREFIXES


def test_rescaled_ticks_and_units():
    scale = 6
    data = {
        'name': 'param',
        'label': 'Parameter',
        'unit': 'V',
        'data': np.array(np.arange(0, 10, 1)*10**scale)
    }
    expected_prefix = _ENGINEERING_PREFIXES[scale]

    ticks_formatter, label = _make_rescaled_ticks_and_units(data)

    assert f"{data['label']} ({expected_prefix}{data['unit']})" == label

    assert '5' == ticks_formatter(5/(10**(-scale)))
    assert '2.12346' == ticks_formatter(2.123456789 / (10 ** (-scale)))
    assert '1' == ticks_formatter(1/(10 ** (-scale)))
