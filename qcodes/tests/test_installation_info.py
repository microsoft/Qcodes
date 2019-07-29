from qcodes.utils.installation_info import _pip_list_parser


def test_pip_list_parser():

    lines = ['Package Version              Location',
             '------- -------------------- ----------------------------',
             'qcodes  0.4.0+186.g6579bf8ae c:\\users\\user\\qcodes']

    packages = _pip_list_parser(lines)

    expected_packages = {'qcodes': {'version': '0.4.0+186.g6579bf8ae',
                                    'location': 'c:\\users\\user\\qcodes'}}

    assert packages == expected_packages
