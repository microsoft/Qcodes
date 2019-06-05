"""
Test module for snapshots of instruments and parameters
"""

from qcodes.tests.instrument_mocks import SnapShotTestInstrument
import pytest


@pytest.mark.parametrize("params,params_to_skip",
                         [(['v1', 'v2', 'v3', 'v4'], ['v1']),
                          (['v1', 'v2', 'v3', 'v4'], ['v2']),
                          (['v1', 'v2', 'v3', 'v4'], ['v3']),
                          (['v1', 'v2', 'v3', 'v4'], ['v4'])])
def test_snapshot_skip_params_update(request, params, params_to_skip):
    """
    Test that params_to_skip_update works as expected, in particular that only
    the parameters given by that variable are skipped.

    This test does not directly call snapshot_base, because this is the way a
    driver will work "in the wild"; the params_to_skip update are baked into
    the driver (in this case passed to the constructor) and the driver only
    calls :meth:`snapshot`
    """

    inst = SnapShotTestInstrument('inst',
                                  params=params,
                                  params_to_skip=params_to_skip)
    request.addfinalizer(inst.close)


    assert list(inst._get_calls.values()) == [0, 0, 0, 0]

    inst.snapshot(update=False)

    assert list(inst._get_calls.values()) == [0, 0, 0, 0]

    inst.snapshot(update=True)

    expected_list = [1, 1, 1, 1]
    expected_list[params.index(params_to_skip[0])] = 0

    assert list(inst._get_calls.values()) == expected_list
