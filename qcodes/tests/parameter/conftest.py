import pytest

NOT_PASSED = 'NOT_PASSED'

@pytest.fixture(params=(True, False, NOT_PASSED))
def snapshot_get(request):
    return request.param


@pytest.fixture(params=(True, False, NOT_PASSED))
def snapshot_value(request):
    return request.param


@pytest.fixture(params=(None, False, NOT_PASSED))
def get_cmd(request):
    return request.param


@pytest.fixture(params=(True, False, NOT_PASSED))
def get_if_invalid(request):
    return request.param


@pytest.fixture(params=(True, False, None, NOT_PASSED))
def update(request):
    return request.param


@pytest.fixture(params=(True, False))
def cache_is_valid(request):
    return request.param
