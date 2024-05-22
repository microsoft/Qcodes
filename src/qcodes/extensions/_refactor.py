from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from textwrap import dedent
from typing import Union, cast

try:
    import libcst as cst
    from libcst import matchers as m
    from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand
except ImportError as er:
    raise ImportError(
        "qcodes-refactor requires that QCoDeS is installed with refactor extra dependencies."
    ) from er


@dataclass
class Extracted:
    name: str | None = None
    docstring: str | None = None
    parameter_class: str | None = None


class AddParameterTransformer(VisitorBasedCodemodCommand):
    """
    This is a tool that will transform `self.add_parameter("name", ...)`
    into the form `self.name: Parameter = self.add_parameter("name", ...)`
    as well as as add a docstring either by extracting the docstring arg
    from add_parameter or by generating a generic docstring.
    This enabled parameters to be typechecked and documented
    """

    def __init__(self, context: CodemodContext) -> None:
        self.annotations: Extracted = Extracted()
        self._arg_num: dict[str, int] = {}
        self._call_stack: list[str] = []
        super().__init__(context=context)

    def visit_Call(self, node: cst.Call) -> None:
        call_name = _get_call_name(node)
        if call_name is not None:
            self._call_stack.append(call_name)
            self._arg_num[call_name] = 0

    def visit_Arg(self, node: cst.Arg) -> None:
        if len(self._call_stack) == 0:
            return
        self._arg_num[self._call_stack[-1]] += 1
        if self._call_stack[-1] != "self.add_parameter":
            return

        first_positional_arg_is_str = self._arg_num[
            self._call_stack[-1]
        ] == 1 and m.matches(
            node,
            matcher=m.Arg(value=m.SimpleString(), keyword=None),
        )
        arg_is_name = m.matches(
            node, m.Arg(keyword=m.Name(value="name"), value=m.SimpleString())
        )

        arg_is_docstring = m.matches(
            node,
            matcher=m.Arg(
                keyword=m.Name(value="docstring"),
                value=m.OneOf(m.SimpleString(), m.ConcatenatedString()),
            ),
        )

        arg_is_docstring_dedent = m.matches(
            node,
            m.Arg(
                keyword=m.Name(value="docstring"),
                value=m.Call(args=[m.Arg(m.SimpleString())]),
            ),
        )

        arg_is_parameter_class = m.matches(
            node,
            matcher=m.Arg(keyword=m.Name(value="parameter_class"), value=m.Name()),
        )

        second_positional_arg_is_str = self._arg_num[
            self._call_stack[-1]
        ] == 2 and m.matches(
            node,
            matcher=m.Arg(value=m.Name(), keyword=None),
        )

        if first_positional_arg_is_str or arg_is_name:
            self.annotations.name = cst.ensure_type(
                node.value, cst.SimpleString
            ).raw_value

        if arg_is_docstring:
            self.annotations.docstring = str(
                cast(
                    Union[cst.SimpleString, cst.ConcatenatedString], node.value
                ).evaluated_value
            )
        if arg_is_docstring_dedent:
            self.annotations.docstring = dedent(
                str(
                    cst.ensure_type(
                        cst.ensure_type(node.value, cst.Call).args[0].value,
                        cst.SimpleString,
                    ).evaluated_value
                )
            ).strip()

        if arg_is_parameter_class or second_positional_arg_is_str:
            self.annotations.parameter_class = cst.ensure_type(
                node.value, cst.Name
            ).value

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        call_name = _get_call_name(updated_node)
        if call_name is not None:
            self._call_stack.pop()
        return updated_node

    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine,
    ) -> cst.FlattenSentinel | cst.SimpleStatementLine:
        annotations = self.annotations
        self.annotations = Extracted()

        if not m.matches(
            updated_node.body[0],
            m.Expr(value=m.Call(func=m.Attribute(attr=m.Name("add_parameter")))),
        ):
            return updated_node

        call_node = cst.ensure_type(
            cst.ensure_type(updated_node.body[0], cst.Expr).value, cst.Call
        )

        if annotations.name is None:
            return updated_node

        if annotations.parameter_class is not None:
            parameter_class = annotations.parameter_class
        else:
            parameter_class = "Parameter"

        if annotations.docstring is not None:
            comment = cst.parse_module(f'"""{annotations.docstring}"""')
        else:
            comment = cst.parse_module(f'"""Parameter {annotations.name}"""')

        stm = cst.parse_statement(
            source=f"self.{annotations.name}: {parameter_class} = x"
        )

        if isinstance(stm.body, cst.BaseSuite) or isinstance(
            comment.body, cst.BaseSuite
        ):
            raise RuntimeError("Unexpected result from parsing code.")

        new_node = stm.body[0]
        new_node = new_node.with_changes(value=call_node)

        return cst.FlattenSentinel(
            [updated_node.with_changes(body=[new_node]), comment.body[0]]
        )


def transform_files_in_folder(folder_path: Path) -> None:
    transformer = AddParameterTransformer(CodemodContext())
    files = glob(f"{folder_path}/**/*.py", recursive=True)
    for filename in files:
        if filename.endswith(".py"):
            try:
                file_name: str = os.path.join(folder_path, filename)
                with open(file_name, encoding="utf-8") as file:
                    source_code = file.read()
                source_tree = cst.parse_module(source_code)
                modified_tree = source_tree.visit(transformer)
                if source_tree.code != modified_tree.code:
                    with open(file_name, "w", encoding="utf-8") as file:
                        print(f"Modified {file_name}")
                        file.write(modified_tree.code)
            except Exception as ex:
                print(f"could not parse {filename}. Got {ex}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="QCoDeS refactor tool",
        description="Perform refactors related to QCoDeS.",
    )
    choices = ["parameter_refactor"]

    parser.add_argument("tool", choices=choices, help="Refactor tool to run.")
    parser.add_argument(
        "path", help="Path to folder where the transform should be run."
    )

    args = parser.parse_args()

    if args.tool == choices[0]:
        path = Path(args.path).absolute()
        transform_files_in_folder(folder_path=path)


def _get_call_name(node: cst.Call) -> str | None:
    if m.matches(node.func, m.Attribute(value=m.Name(), attr=m.Name())):
        my_class = cst.ensure_type(
            cst.ensure_type(node.func, cst.Attribute).value, cst.Name
        ).value
        my_attr = cst.ensure_type(
            cst.ensure_type(node.func, cst.Attribute).attr, cst.Name
        ).value

        func_name = f"{my_class}.{my_attr}"
    elif m.matches(node.func, m.Name()):
        func_name = cst.ensure_type(node.func, cst.Name).value
    else:
        return None

    return func_name
