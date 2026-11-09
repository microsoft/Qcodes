import inspect
from typing import TYPE_CHECKING, Any

import pytest

from qcodes.instrument import InstrumentBase
from qcodes.sphinx_extensions.github_source_links import (
    find_source,
    github_ref,
    make_linkcode_resolve,
)

if TYPE_CHECKING:
    from collections.abc import Callable

INSTRUMENT_BASE_PATH = "qcodes/instrument/instrument_base.py"


def _lines(obj: "Callable[..., Any] | None") -> tuple[int, int]:
    """The first and last line of the definition of obj including decorators."""
    assert obj is not None
    lines, start = inspect.getsourcelines(obj)
    return start, start + len(lines) - 1


@pytest.mark.parametrize(
    ("version", "ref"),
    [
        pytest.param("0.62.0", "v0.62.0", id="release"),
        pytest.param("0.12.0rc1", "v0.12.0rc1", id="release_candidate"),
        pytest.param("0.63.0.dev12+main.g76c540bef", "76c540bef", id="dev"),
        pytest.param(
            "0.63.0.dev12+my.branch.g76c540bef.dirty", "76c540bef", id="dev_dirty"
        ),
        pytest.param(
            "0.63.0.dev12+main.g0123456789abcdef0123456789abcdef01234567",
            "0123456789abcdef0123456789abcdef01234567",
            id="dev_full_hash",
        ),
        pytest.param(
            "0.63.0.dev12+gadget.g76c540bef", "76c540bef", id="branch_starting_with_g"
        ),
        # the default-version that versioningit uses when git is not available
        pytest.param("0.63.0dev+Unknown", "main", id="no_git"),
        pytest.param("0.63.0.dev12", "main", id="dev_without_commit"),
    ],
)
def test_github_ref(version: str, ref: str) -> None:
    assert github_ref(version) == ref


def test_find_source_of_class() -> None:
    assert find_source("qcodes.instrument", "InstrumentBase") == (
        INSTRUMENT_BASE_PATH,
        *_lines(InstrumentBase),
    )


def test_find_source_of_method() -> None:
    assert find_source(
        "qcodes.instrument", "InstrumentBase.print_readable_snapshot"
    ) == (INSTRUMENT_BASE_PATH, *_lines(InstrumentBase.print_readable_snapshot))


def test_find_source_of_inherited_method_is_in_base_class() -> None:
    assert find_source(
        "qcodes.instrument", "VisaInstrument.print_readable_snapshot"
    ) == (INSTRUMENT_BASE_PATH, *_lines(InstrumentBase.print_readable_snapshot))


def test_find_source_of_property() -> None:
    assert find_source("qcodes.instrument", "InstrumentBase.full_name") == (
        INSTRUMENT_BASE_PATH,
        *_lines(InstrumentBase.full_name.fget),
    )


@pytest.mark.parametrize(
    ("modname", "fullname"),
    [
        pytest.param("qcodes.instrument", "VisaInstrument.default_timeout", id="attr"),
        pytest.param("qcodes.instrument", "InstrumentBase.not_an_attr", id="missing"),
        pytest.param("qcodes.not_a_module", "InstrumentBase", id="missing_module"),
        pytest.param("json", "dumps", id="not_in_qcodes"),
    ],
)
def test_find_source_without_source_in_qcodes(modname: str, fullname: str) -> None:
    assert find_source(modname, fullname) is None


def test_linkcode_resolve() -> None:
    linkcode_resolve = make_linkcode_resolve("microsoft/Qcodes", "0.62.0")
    start, end = _lines(InstrumentBase.print_readable_snapshot)

    url = linkcode_resolve(
        "py",
        {"module": "qcodes.instrument", "fullname": "VisaInstrument.print_readable_snapshot"},
    )

    assert url == (
        "https://github.com/microsoft/Qcodes/blob/v0.62.0/src/"
        f"{INSTRUMENT_BASE_PATH}#L{start}-L{end}"
    )


def test_linkcode_resolve_without_source() -> None:
    linkcode_resolve = make_linkcode_resolve("microsoft/Qcodes", "0.62.0")

    assert linkcode_resolve("py", {"module": "json", "fullname": "dumps"}) is None
    assert linkcode_resolve("c", {"names": "qcodes_function"}) is None
