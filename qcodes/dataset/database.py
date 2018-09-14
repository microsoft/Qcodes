# high-level interface to the database

from os.path import expanduser

from qcodes.dataset.sqlite_base import connect as _connect
from qcodes.dataset.sqlite_base import init_db as _init_db
import qcodes.config


def get_DB_location() -> str:
    return expanduser(qcodes.config["core"]["db_location"])


def get_DB_debug() -> bool:
    return bool(qcodes.config["core"]["db_debug"])


def initialise_database() -> None:
    """
    Initialise a database in the location specified by the config object
    If the database already exists, nothing happens. The database is
    created with or upgraded to the newest version

    Args:
        config: An instance of the config object
    """
    conn = _connect(get_DB_location(), get_DB_debug())
    # init is actually idempotent so it's safe to always call!
    _init_db(conn)
    conn.close()
    del conn


def initialise_or_create_database_at(db_file_with_abs_path: str) -> None:
    """
    This function sets up QCoDeS to refer to the given database file. If the
    database file does not exist, it will be initiated.

    Args:
        db_file_with_abs_path
            Database file name with absolute path, for example
            ``C:\\mydata\\majorana_experiments.db``
    """
    qcodes.config.core.db_location = db_file_with_abs_path
    initialise_database()
