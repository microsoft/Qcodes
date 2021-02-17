from typing import Tuple, Union, Dict, Optional, List, Iterable, cast

from pathlib import Path
import gc
import ast

from sqlite3 import DatabaseError

from qcodes.dataset.guids import validate_guid_format
from qcodes.dataset.sqlite.queries import get_guids_from_run_spec
from qcodes.dataset.sqlite.database import connect


def guids_from_dbs(
    db_paths: Iterable[Path],
) -> Tuple[Dict[Path, List[str]], Dict[str, Path]]:
    """
    Extract all guids from the supplied database paths.

    Args:
        db_paths: Path or str or directory where to search

    Returns:
        Tuple of Dictionary mapping paths to lists of guids as strings
        and Dictionary mapping guids to db paths.
    """
    dbdict = {}
    for p in db_paths:
        try:
            conn = connect(str(p))
            dbdict[p] = get_guids_from_run_spec(conn)
        except (RuntimeError, DatabaseError) as e:
            print(e)
        finally:
            conn.close()
            gc.collect()
    guiddict = {}
    for dbpath, guids in dbdict.items():
        guiddict.update({guid: dbpath for guid in guids})
    return dbdict, guiddict


def guids_from_dir(
    basepath: Union[Path, str]
) -> Tuple[Dict[Path, List[str]], Dict[str, Path]]:
    """
    Recursively find all db files under basepath and extract guids.

    Args:
        basepath: Path or str or directory where to search

    Returns:
        Tuple of Dictionary mapping paths to lists of guids as strings
        and Dictionary mapping guids to db paths.
    """
    return guids_from_dbs(Path(basepath).glob("**/*.db"))


def guids_from_list_str(s: str) -> Optional[Tuple[str, ...]]:
    """
    Get tuple of guids from a python/json string representation of a list.

    Extracts the guids from a string representation of a list, tuple,
    or set of guids or a single guid.

    Args:
        s: input string

    Returns:
        Extracted guids as a tuple of strings.
        If a provided string does not match the format, `None` will be returned.
        For an empty list/tuple/set or empty string an empty tuple is returned.

    Examples:
        >>> guids_from_str(
        "['07fd7195-c51e-44d6-a085-fa8274cf00d6', \
          '070d7195-c51e-44d6-a085-fa8274cf00d6']")
        will return
        ('07fd7195-c51e-44d6-a085-fa8274cf00d6',
        '070d7195-c51e-44d6-a085-fa8274cf00d6')
    """
    if s == "":
        return tuple()

    try:
        validate_guid_format(s)
        return (s,)
    except ValueError:
        pass

    try:
        parsed_expression = ast.parse(s, mode='eval')
    except SyntaxError:
        return None

    if not hasattr(parsed_expression, 'body'):
        return None

    parsed = cast(ast.Expression, parsed_expression).body

    if isinstance(parsed, ast.Str):
        if len(parsed.s) > 0:
            return (parsed.s,)
        else:
            return tuple()

    if not isinstance(parsed, (ast.List, ast.Tuple, ast.Set)):
        return None

    if not all(isinstance(e, ast.Str) for e in parsed.elts):
        return None

    str_elts = cast(Tuple[ast.Str, ...], tuple(parsed.elts))
    return tuple(s.s for s in str_elts)
