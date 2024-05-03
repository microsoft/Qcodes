from textwrap import dedent

import libcst as cst
from libcst.codemod import CodemodContext

from qcodes.extensions._refactor import AddParameterTransformer


def transform_code(code: str) -> str:
    transformer = AddParameterTransformer(CodemodContext())
    module = cst.parse_module(code)
    new_module = module.visit(transformer)
    return new_module.code


def test_transform_add_parameter_basic() -> None:
    code = dedent(
        """
        from qcodes.instrument_drivers.rohde_schwarz.RTO1000 import ScopeTrace
        self.add_parameter(
            "cool_time",
            label="Cooling Time",
            unit="s",
            get_cmd="PS:CTIME?",
            get_parser=int,
            set_cmd="CONF:PS:CTIME {}",
            vals=Ints(5, 3600),
        )
        """
    )
    output = transform_code(code)
    output_lines = output.split("\n")
    assert output_lines[2].startswith("self.cool_time: Parameter = self.add_parameter")
    assert output_lines[-2] == '"""Parameter cool_time"""'


def test_transform_add_parameter_kwarg_class() -> None:
    code = dedent(
        """
        from qcodes.instrument_drivers.rohde_schwarz.RTO1000 import ScopeTrace
        self.add_parameter(
            "cool_time",
            label="Cooling Time",
            unit="s",
            get_cmd="PS:CTIME?",
            get_parser=int,
            set_cmd="CONF:PS:CTIME {}",
            vals=Ints(5, 3600),
            parameter_class=ScopeTrace,

        )
        """
    )
    output = transform_code(code)
    output_lines = output.split("\n")
    assert output_lines[2].startswith("self.cool_time: ScopeTrace = self.add_parameter")
    assert output_lines[-2] == '"""Parameter cool_time"""'


def test_transform_add_parameter_docstring() -> None:
    code = dedent(
        """
        from qcodes.instrument_drivers.rohde_schwarz.RTO1000 import ScopeTrace
        self.add_parameter(
            "cool_time",
            label="Cooling Time",
            unit="s",
            get_cmd="PS:CTIME?",
            get_parser=int,
            set_cmd="CONF:PS:CTIME {}",
            vals=Ints(5, 3600),
            docstring="This parameter has a docstring",
        )
        """
    )
    output = transform_code(code)
    output_lines = output.split("\n")
    assert output_lines[2].startswith("self.cool_time: Parameter = self.add_parameter")
    assert output_lines[-2] == '"""This parameter has a docstring"""'


def test_transform_add_parameter_name_kwarg() -> None:
    code = dedent(
        """
        from qcodes.instrument_drivers.rohde_schwarz.RTO1000 import ScopeTrace
        self.add_parameter(
            name="cool_time",
            label="Cooling Time",
            unit="s",
            get_cmd="PS:CTIME?",
            get_parser=int,
            set_cmd="CONF:PS:CTIME {}",
            vals=Ints(5, 3600),
        )
        """
    )
    output = transform_code(code)
    output_lines = output.split("\n")
    assert output_lines[2].startswith("self.cool_time: Parameter = self.add_parameter")
    assert output_lines[-2] == '"""Parameter cool_time"""'


def test_transform_add_parameter_param_class_positional() -> None:
    code = dedent(
        """
        from qcodes.instrument_drivers.rohde_schwarz.RTO1000 import ScopeTrace
        self.add_parameter(
            "cool_time",
            ScopeTrace,
            label="Cooling Time",
            unit="s",
            get_cmd="PS:CTIME?",
            get_parser=int,
            set_cmd="CONF:PS:CTIME {}",
            vals=Ints(5, 3600),
        )
        """
    )
    output = transform_code(code)
    output_lines = output.split("\n")
    assert output_lines[2].startswith("self.cool_time: ScopeTrace = self.add_parameter")
    assert output_lines[-2] == '"""Parameter cool_time"""'


def test_transform_add_parameter_multiline_docstring() -> None:
    code = dedent(
        """
        from qcodes.instrument_drivers.rohde_schwarz.RTO1000 import ScopeTrace
        self.add_parameter(
            "cool_time",
            ScopeTrace,
            label="Cooling Time",
            unit="s",
            get_cmd="PS:CTIME?",
            get_parser=int,
            set_cmd="CONF:PS:CTIME {}",
            vals=Ints(5, 3600),
            docstring = (
                "This is the first line "
                "This is the second line"
            )
        )
        """
    )
    output = transform_code(code)
    output_lines = output.split("\n")
    assert output_lines[2].startswith("self.cool_time: ScopeTrace = self.add_parameter")
    assert output_lines[-2] == '"""This is the first line This is the second line"""'


def test_transform_add_parameter_dedent():
    code = dedent(
        """
            self.add_parameter(
                name="sweep_auto_abort",
                set_cmd=self._set_sweep_auto_abort,
                get_cmd=self._get_sweep_auto_abort,
                set_parser=constants.Abort,
                get_parser=constants.Abort,
                vals=vals.Enum(*list(constants.Abort)),
                initial_cache_value=constants.Abort.ENABLED,
                docstring=textwrap.dedent(\"\"\"
                                    enables or disables the automatic abort function
                                    for the CV (DC bias) sweep measurement (MM18) and
                                    the pulsed bias sweep measurement (MM20). The
                                    automatic abort function stops the measurement
                                    when one of the following conditions occurs:
                                        - NULL loop unbalance condition
                                        - IV amplifier saturation condition
                                        - Overflow on the AD converter
                                    \"\"\"),
                                    )
                """
    )
    output = transform_code(code)
    output_lines = output.split("\n")
    assert output_lines[-2] == '    - Overflow on the AD converter"""'
