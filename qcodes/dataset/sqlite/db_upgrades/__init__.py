import logging
from functools import wraps
from typing import Dict, Callable

from qcodes.dataset.sqlite.connection import ConnectionPlus
from qcodes.dataset.sqlite.db_upgrades.version import get_user_version, \
    set_user_version


log = logging.getLogger(__name__)


# Functions decorated as 'upgrader' are inserted into this dict
# The newest database version is thus determined by the number of upgrades
# in this module
# The key is the TARGET VERSION of the upgrade, i.e. the first key is 1
_UPGRADE_ACTIONS: Dict[int, Callable] = {}


def _latest_available_version() -> int:
    """Return latest available database schema version"""
    return len(_UPGRADE_ACTIONS)


def upgrader(func: Callable[[ConnectionPlus], None]):
    """
    Decorator for database version upgrade functions. An upgrade function
    must have the name `perform_db_upgrade_N_to_M` where N = M-1. For
    simplicity, an upgrade function must take a single argument of type
    `ConnectionPlus`. The upgrade function must either perform the upgrade
    and return (no return values allowed) or fail to perform the upgrade,
    in which case it must raise a RuntimeError. A failed upgrade must be
    completely rolled back before the RuntimeError is raises.

    The decorator takes care of logging about the upgrade and managing the
    database versioning.
    """
    name_comps = func.__name__.split('_')
    if not len(name_comps) == 6:
        raise NameError('Decorated function not a valid upgrader. '
                        'Must have name "perform_db_upgrade_N_to_M"')
    if not ''.join(name_comps[:3]+[name_comps[4]]) == 'performdbupgradeto':
        raise NameError('Decorated function not a valid upgrader. '
                        'Must have name "perform_db_upgrade_N_to_M"')
    from_version = int(name_comps[3])
    to_version = int(name_comps[5])

    if not to_version == from_version+1:
        raise ValueError(f'Invalid upgrade versions in function name: '
                         f'{func.__name__}; upgrade from version '
                         f'{from_version} to version {to_version}.'
                         ' Can only upgrade from version N'
                         ' to version N+1')

    @wraps(func)
    def do_upgrade(conn: ConnectionPlus) -> None:

        log.info(f'Starting database upgrade version {from_version} '
                 f'to {to_version}')

        start_version = get_user_version(conn)
        if start_version != from_version:
            log.info(f'Skipping upgrade {from_version} -> {to_version} as'
                     f' current database version is {start_version}.')
            return

        # This function either raises or returns
        func(conn)

        set_user_version(conn, to_version)
        log.info(f'Succesfully performed upgrade {from_version} '
                 f'-> {to_version}')

    _UPGRADE_ACTIONS[to_version] = do_upgrade

    return do_upgrade


def perform_db_upgrade(conn: ConnectionPlus, version: int = -1) -> None:
    """
    This is intended to perform all upgrades as needed to bring the
    db from version 0 to the most current version (or the version specified).
    All the perform_db_upgrade_X_to_Y functions must raise if they cannot
    upgrade and be a NOOP if the current version is higher than their target.

    Args:
        conn: object for connection to the database
        version: Which version to upgrade to. We count from 0. -1 means
          'newest version'
    """
    version = _latest_available_version() if version == -1 else version

    current_version = get_user_version(conn)
    if current_version < version:
        log.info("Commencing database upgrade")
        for target_version in sorted(_UPGRADE_ACTIONS)[:version]:
            _UPGRADE_ACTIONS[target_version](conn)
