"""
Test module for snapshots of instruments and parameters
"""

from qcodes.tests.instrument_mocks import SnapShotTestInstrument
import pytest


@pytest.mark.parametrize("params,params_to_skip",
                         [(['v1', 'v2', 'v3', 'v4'], ['v1']),
                          (['v1', 'v2', 'v3', 'v4'], ['v2']),
                          (['v1', 'v2', 'v3', 'v4'], ['v3']),
                          (['v1', 'v2', 'v3', 'v4'], ['v4']),
                          (['v1', 'v2', 'v3', 'v4'], [])])
def test_snapshot_skip_params_update(request, params, params_to_skip):
    """
    Test that params_to_skip_update works as expected, in particular that only
    the parameters given by that variable are skipped.

    This test does not directly call snapshot_base, because this is the way a
    driver will work "in the wild"; the params_to_skip update are baked into
    the driver (in this case passed to the constructor) and the driver only
    calls :meth:`snapshot`
    """

    inst = SnapShotTestInstrument('snapshot_inst_1',
                                  params=params,
                                  params_to_skip=params_to_skip)
    request.addfinalizer(inst.close)

    assert list(inst._get_calls.values()) == [0, 0, 0, 0]

    inst.snapshot(update=False)

    assert list(inst._get_calls.values()) == [0, 0, 0, 0]

    inst.snapshot(update=True)

    expected_list = [1, 1, 1, 1]
    if params_to_skip:
        expected_list[params.index(params_to_skip[0])] = 0

    assert list(inst._get_calls.values()) == expected_list


@pytest.mark.parametrize("params,params_to_exclude",
                         [(['v1', 'v2', 'v3', 'v4'], ['v1']),
                          (['v1', 'v2', 'v3', 'v4'], ['v2']),
                          (['v1', 'v2', 'v3', 'v4'], ['v3']),
                          (['v1', 'v2', 'v3', 'v4'], ['v4']),
                          (['v1', 'v2', 'v3', 'v4'], ['v4']),
                          (['v1', 'v2', 'v3', 'v4'], ['v1', 'v2']),
                          (['v1', 'v2', 'v3', 'v4'], [])])
def test_snapshot_exclude_params(request, params, params_to_exclude):
    """
    Test that params_to_exclude works as expected, in particular that only
    the parameters given by that variable are excluded from the snapshot.
    This test does not directly call snapshot_base, because this is the way a
    driver will work "in the wild"; the params_to_exclude update are baked into
    the driver (in this case passed to the constructor) and the driver only
    calls :meth:`snapshot`
    """

    inst = SnapShotTestInstrument('snapshot_inst_2',
                                  params=params,
                                  params_to_skip=[])
    request.addfinalizer(inst.close)

    params.insert(0, "IDN")  # Is added by default to a instrument

    for param_exl in params_to_exclude:
        inst.parameters[param_exl].snapshot_exclude = True

    snap = inst.snapshot()
    snap_params = [x for x in snap["parameters"]]
    params = [x for x in params if x not in params_to_exclude]

    assert all(elem not in snap_params for elem in params_to_exclude), \
        f"Parameter(s) is not excluded from the snapshot. expected: " \
        f"{params}, actual: {snap_params}"\

    assert snap_params == params, f"Snapshot does not contain the expected " \
        f"parameters expected: {params}, actual: {snap_params}"
