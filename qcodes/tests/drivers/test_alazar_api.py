import pytest

from qcodes.instrument_drivers.AlazarTech.ATS import AlazarTech_ATS


@pytest.fixture
def alazar():
    alazar = AlazarTech_ATS('alazar', system_id=2)
    yield alazar
    alazar.close()


def test_find_boards():
    boards = AlazarTech_ATS.find_boards()
    assert isinstance(boards, list)


def test_init_alazar(alazar):
    print(alazar.get_idn())
