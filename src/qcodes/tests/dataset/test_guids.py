import random
import re
import time
from uuid import uuid4

import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings

import qcodes as qc
from qcodes.dataset.guids import (
    filter_guids_by_parts,
    generate_guid,
    parse_guid,
    set_guid_location_code,
    set_guid_work_station_code,
    validate_guid_format,
)


@pytest.fixture(name="seed_random")
def _make_seed_random():
    state = random.getstate()
    random.seed(a=0)
    try:
        yield
    finally:
        random.setstate(state)


@settings(max_examples=50, deadline=1000)
@given(
    loc=hst.integers(0, 0xFF),
    stat=hst.integers(0, 0xFFFF),
    smpl=hst.integers(0, 0xFF_FFF_FFF),
)
def test_generate_legacy_guid(loc, stat, smpl) -> None:
    """
    Generate a guid in the legacy format (with an assigned sample name)
    and verify that it is parsed
    """
    # update config to generate a particular guid. Read it back to verify
    cfg = qc.config
    cfg["GUID_components"]["location"] = loc
    cfg["GUID_components"]["work_station"] = stat
    cfg["GUID_components"]["sample"] = smpl
    cfg["GUID_components"]["GUID_type"] = "explicit_sample"

    if smpl in (0, 0xAA_AAA_AAA):
        guid = generate_guid()
    else:
        with pytest.warns(
            expected_warning=Warning,
            match=re.escape("Setting a non default GUID_components.sample"),
        ):
            guid = generate_guid()

    gen_time = int(np.round(time.time() * 1000))

    comps = parse_guid(guid)

    if smpl == 0:
        smpl = 0xAA_AAA_AAA

    assert comps["location"] == loc
    assert comps["work_station"] == stat
    assert comps["sample"] == smpl
    assert comps["time"] - gen_time < 2


@settings(max_examples=50, deadline=1000)
@given(
    loc=hst.integers(0, 0xFF),
    stat=hst.integers(0, 0xFFFF),
    smpl=hst.integers(0, 0xFF_FFF_FFF),
)
def test_generate_guid(loc, stat, smpl) -> None:
    """
    Generate a guid and verify that it is parsed or raises
    if a sample name is given
    """
    # update config to generate a particular guid. Read it back to verify
    cfg = qc.config
    cfg["GUID_components"]["location"] = loc
    cfg["GUID_components"]["work_station"] = stat
    cfg["GUID_components"]["sample"] = smpl

    if smpl in (0, 0xAA_AAA_AAA):
        guid = generate_guid()
        gen_time = int(np.round(time.time() * 1000))

        comps = parse_guid(guid)

        if smpl == 0:
            smpl = 0xAA_AAA_AAA

        assert comps["location"] == loc
        assert comps["work_station"] == stat
        # assert comps["sample"] == smpl
        assert comps["time"] - gen_time < 2
    else:
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "QCoDeS is configured to disregard GUID_components.sample "
                "from config file but this"
            ),
        ):
            _ = generate_guid()


@settings(max_examples=50, deadline=None,
          suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(loc=hst.integers(-10, 350))
def test_set_guid_location_code(loc, monkeypatch) -> None:
    monkeypatch.setattr('builtins.input', lambda x: str(loc))
    orig_cfg = qc.config
    original_loc = orig_cfg["GUID_components"]["location"]
    set_guid_location_code()

    cfg = qc.config

    if 257 > loc > 0:
        assert cfg["GUID_components"]["location"] == loc
    else:
        assert cfg["GUID_components"]["location"] == original_loc


@settings(max_examples=50, deadline=1000,
          suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(ws=hst.integers(-10, 17000000))
def test_set_guid_workstation_code(ws, monkeypatch) -> None:
    monkeypatch.setattr('builtins.input', lambda x: str(ws))

    orig_cfg = qc.config
    original_ws = orig_cfg["GUID_components"]["work_station"]

    set_guid_work_station_code()

    cfg = qc.config

    if 16777216 > ws > 0:
        assert cfg["GUID_components"]["work_station"] == ws
    else:
        assert cfg["GUID_components"]["work_station"] == original_ws


@settings(max_examples=50, deadline=1000)
@given(
    locs=hst.lists(hst.integers(0, 0xFF), min_size=2, max_size=2, unique=True),
    stats=hst.lists(hst.integers(0, 0xFFFF), min_size=2, max_size=2, unique=True),
    smpls=hst.lists(hst.integers(0, 0xFF_FFF_FFF), min_size=2, max_size=2, unique=True),
)
def test_filter_guid(locs, stats, smpls) -> None:
    def make_test_guid(cfg, loc: int, smpl: int, stat: int):
        cfg["GUID_components"]["location"] = loc
        cfg["GUID_components"]["work_station"] = stat
        cfg["GUID_components"]["sample"] = smpl
        # even thou setting the sample name is no longer supported
        # by default it makes sense to still be able to filter
        # old dataset on that since they already exist
        cfg["GUID_components"]["GUID_type"] = "explicit_sample"

        if smpl in (0, 0xAAAAAAAA):
            guid = generate_guid()
        else:
            with pytest.warns(
                expected_warning=Warning,
                match=re.escape("Setting a non default GUID_components.sample"),
            ):
                guid = generate_guid()

        gen_time = int(np.round(time.time() * 1000))

        comps = parse_guid(guid)

        assert comps['location'] == loc
        assert comps['work_station'] == stat
        assert comps['sample'] == smpl
        assert comps['time'] - gen_time < 2

        return guid

    guids = []
    cfg = qc.config

    corrected_smpls = [smpl if smpl != 0 else int("a" * 8, base=16) for smpl in smpls]
    # there is a possibility that we could generate 0 and 2863311530, which
    # are considered equivalent since int('a' * 8, base=16) == 2863311530.
    # We want unique samples, so we exclude this case.
    assume(corrected_smpls[0] != corrected_smpls[1])

    # first we generate a guid that we are going to match against
    guids.append(make_test_guid(cfg, locs[0], corrected_smpls[0], stats[0]))

    # now generate some guids that will not match because one of the
    # components changed
    guids.append(make_test_guid(cfg, locs[1], corrected_smpls[0], stats[0]))
    guids.append(make_test_guid(cfg, locs[0], corrected_smpls[1], stats[0]))
    guids.append(make_test_guid(cfg, locs[0], corrected_smpls[0], stats[1]))

    assert len(guids) == 4

    # first filter on all parts. This should give exactly one matching guid
    filtered_guids = filter_guids_by_parts(
        guids, location=locs[0], sample_id=corrected_smpls[0], work_station=stats[0]
    )

    assert len(filtered_guids) == 1
    assert filtered_guids[0] == guids[0]

    # now filter on 2 components
    filtered_guids = filter_guids_by_parts(
        guids, location=locs[0], sample_id=corrected_smpls[0]
    )
    assert len(filtered_guids) == 2
    assert filtered_guids[0] == guids[0]
    assert filtered_guids[1] == guids[3]

    filtered_guids = filter_guids_by_parts(
        guids, location=locs[0], work_station=stats[0]
    )
    assert len(filtered_guids) == 2
    assert filtered_guids[0] == guids[0]
    assert filtered_guids[1] == guids[2]

    filtered_guids = filter_guids_by_parts(
        guids, sample_id=corrected_smpls[0], work_station=stats[0]
    )
    assert len(filtered_guids) == 2
    assert filtered_guids[0] == guids[0]
    assert filtered_guids[1] == guids[1]

    # now filter on 1 component
    filtered_guids = filter_guids_by_parts(guids, location=locs[0])
    assert len(filtered_guids) == 3
    assert filtered_guids[0] == guids[0]
    assert filtered_guids[1] == guids[2]
    assert filtered_guids[2] == guids[3]

    filtered_guids = filter_guids_by_parts(guids, work_station=stats[0])
    assert len(filtered_guids) == 3
    assert filtered_guids[0] == guids[0]
    assert filtered_guids[1] == guids[1]
    assert filtered_guids[2] == guids[2]

    filtered_guids = filter_guids_by_parts(
        guids,
        sample_id=corrected_smpls[0],
    )
    assert len(filtered_guids) == 3
    assert filtered_guids[0] == guids[0]
    assert filtered_guids[1] == guids[1]
    assert filtered_guids[2] == guids[3]


def test_validation() -> None:
    valid_guid = str(uuid4())
    validate_guid_format(valid_guid)

    with pytest.raises(ValueError):
        validate_guid_format(valid_guid[1:])


@pytest.mark.usefixtures("seed_random")
def test_random_sample_guid() -> None:

    cfg = qc.config
    cfg["GUID_components"]["GUID_type"] = "random_sample"

    expected_guid_prefixes = ["d82c07ce", "629f6fbf", "c2094cad"]
    for expected_guid_prefix in expected_guid_prefixes:
        guid = generate_guid()
        assert guid.split("-")[0] == expected_guid_prefix


def test_random_sample_and_sample_int_in_guid_raises() -> None:

    cfg = qc.config
    cfg["GUID_components"]["GUID_type"] = "random_sample"

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "QCoDeS is configured to disregard GUID_components.sample from config"
        ),
    ):
        generate_guid(sampleint=10)


def test_sample_int_in_guid_warns_with_old_config() -> None:
    cfg = qc.config
    cfg["GUID_components"]["GUID_type"] = "explicit_sample"
    with pytest.warns(
        expected_warning=Warning,
        match=re.escape("Setting a non default GUID_components.sample"),
    ):
        generate_guid(sampleint=10)


def test_sample_int_in_guid_raises() -> None:
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "QCoDeS is configured to disregard GUID_components.sample "
            "from config file but this"
        ),
    ):
        generate_guid(sampleint=10)
