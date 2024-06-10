from __future__ import annotations

import argparse
import os
from ast import literal_eval
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from textwrap import dedent

try:
    import libcst as cst
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

        match node:
            case cst.Arg(
                value=cst.SimpleString(e_value), keyword=None
            ) if self._arg_num[self._call_stack[-1]]:
                # first positional arg is str
                self.annotations.name = e_value.strip("\"'")
            case cst.Arg(
                keyword=cst.Name(value="name"), value=cst.SimpleString(e_value)
            ):
                # arg is name
                self.annotations.name = e_value.strip("\"'")
            case cst.Arg(
                keyword=cst.Name("docstring"), value=cst.SimpleString(e_value)
            ):
                # arg is docstring
                self.annotations.docstring = literal_eval(e_value)
            case cst.Arg(
                keyword=cst.Name("docstring"),
                value=e_value,
            ) if isinstance(e_value, cst.ConcatenatedString):
                # arg is docstring concatenated
                self.annotations.docstring = str(e_value.evaluated_value)
            case cst.Arg(
                keyword=cst.Name("docstring"),
                value=cst.Call(args=[cst.Arg(cst.SimpleString(e_value))]),
            ):
                # arg is docstring dedent
                self.annotations.docstring = dedent(literal_eval(e_value)).strip()
            case cst.Arg(value=cst.Name(e_value), keyword=None) if self._arg_num[
                self._call_stack[-1]
            ] == 2:
                # second positional arg is str
                self.annotations.parameter_class = e_value
            case cst.Arg(keyword=cst.Name("parameter_class"), value=cst.Name(e_value)):
                # arg is parameter class
                self.annotations.parameter_class = e_value

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

        match updated_node.body[0]:
            case cst.Expr(
                value=cst.Call(func=cst.Attribute(attr=cst.Name("add_parameter")))
            ):
                return self._create_updated_node(annotations, updated_node)
            case _:
                return updated_node

    @staticmethod
    def _create_updated_node(
        annotations: Extracted, updated_node: cst.SimpleStatementLine
    ) -> (
        cst.SimpleStatementLine
        | cst.FlattenSentinel[cst.SimpleStatementLine | cst.BaseCompoundStatement]
    ):
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
    func_name: str | None
    match node:
        case cst.Call(
            func=cst.Attribute(value=cst.Name(my_class), attr=cst.Name(my_attr))
        ):
            func_name = f"{my_class}.{my_attr}"
        case cst.Call(cst.Name(e_value)):
            func_name = e_value
        case _:
            func_name = None
    return func_name
