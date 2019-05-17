"""
Code of this module has been moved to `.sqlite.database`. This module now
only re-imports the functions which it used to contain, for backwards
compatibility. Do not import functions from this module because it will be
removed soon.
"""

from .sqlite.database import get_DB_debug, get_DB_location, \
    initialise_database, initialise_or_create_database_at, path_to_dbfile
