"""CLI tool for generating Alazar ATS API module

The tool uses a `mako` template to generate a python module of tools which are
related to Alazar ATS API. The user of template avoids manual code 
duplication.
"""
from pathlib import Path
import argparse
import textwrap

from mako.template import Template

from qcodes.instrument_drivers import AlazarTech


TEMPLATE_FILE_NAME = 'ats_api.mako_template'
TARGET_MODULE_FILE_NAME = 'ats_api.py'

_target_module_dir = Path(AlazarTech.__file__).parent
TEMPLATE_FILE = _target_module_dir / TEMPLATE_FILE_NAME

TARGET_FILE = _target_module_dir / TARGET_MODULE_FILE_NAME


def generate_ats_api_module(template_file: Path = TEMPLATE_FILE) -> str:
    if not template_file.exists():
        raise FileNotFoundError(template_file)

    template = Template(filename=str(template_file),
                        strict_undefined=True)
    code = template.render()

    return code


def write_code_to_file(code: str,
                       target_file: Path = TARGET_FILE,
                       overwrite: bool = False
                       ) -> None:
    if target_file.exists() and not overwrite:
        raise FileExistsError(target_file)

    with open(target_file, 'w') as f:
        f.write(code)

    print('Done!')


def parse_args() -> dict:
    description = textwrap.dedent(f"""
        Generate Alazar ATS API module from a template file.

        Template:
            {TEMPLATE_FILE}
        Generated module:
            {TARGET_FILE}""")

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-w', '--overwrite',
                        dest='overwrite',
                        action='store_true',
                        help='force overwrite Alazar ATS API module file '
                             'if exists')

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    code = generate_ats_api_module()
    write_code_to_file(code, **args)
