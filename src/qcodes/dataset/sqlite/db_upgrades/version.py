from __future__ import annotations

from qcodes.dataset.sqlite.connection import ConnectionPlusPlus, atomic_transaction
from qcodes.dataset.sqlite.query_helpers import one


def get_user_version(conn: ConnectionPlusPlus) -> int:
    curr = atomic_transaction(conn, "PRAGMA user_version")
    res = one(curr, 0)
    return res


def set_user_version(conn: ConnectionPlusPlus, version: int) -> None:
    atomic_transaction(conn, f"PRAGMA user_version({version})")
