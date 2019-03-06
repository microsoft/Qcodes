from pathlib import Path
from difflib import unified_diff

import pytest
from mako.template import Template

from qcodes.instrument_drivers.AlazarTech._gen_ats_api import \
    generate_ats_api_module, TARGET_FILE, TARGET_MODULE_FILE_NAME


def test_alazar_api_consistent_with_template():
    generated_code = generate_ats_api_module()
    generated_code_lines = generated_code.split("\n")
    
    with open(TARGET_FILE) as ff:
        existing_code_lines = ff.read().split("\n")

    diff_gen = unified_diff(existing_code_lines,
                            generated_code_lines, 
                            fromfile=TARGET_MODULE_FILE_NAME, 
                            tofile='Generated_in_test',
                            lineterm='')
    diff_lines = list(diff_gen)

    if len(diff_lines) > 0:
        msg = 'Files differ:\n' + '\n'.join(diff_lines)
        pytest.fail(msg=msg)
