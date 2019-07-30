import qcodes.utils.installation_info as ii
import qcodes as qc


def test_pip_list_parser():

    lines = ['Package Version              Location',
             '------- -------------------- ----------------------------',
             'qcodes  0.4.0+186.g6579bf8ae c:\\users\\user\\qcodes']

    packages = ii._pip_list_parser(lines)

    expected_packages = {'qcodes': {'version': '0.4.0+186.g6579bf8ae',
                                    'location': 'c:\\users\\user\\qcodes'}}

    assert packages == expected_packages


# The get_* functions from installation_info are hard to meaningfully test,
# but we can at least test that they execute without errors


def test_get_qcodes_version():
    assert ii.get_qcodes_version() == qc.version.__version__


def test_get_qcodes_requirements():
    reqs = ii.get_qcodes_requirements()

    assert type(reqs) == list
    assert len(reqs) > 0


def test_get_qcodes_requirements_versions():
    req_vs = ii.get_qcodes_requirements_versions()

    assert type(req_vs) == dict
    assert len(req_vs) > 0
