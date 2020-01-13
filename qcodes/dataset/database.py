"""
Code of this module has been moved to `.sqlite.database`. This module now
only re-imports the functions which it used to contain, for backwards
compatibility. Do not import functions from this module because it will be
removed soon.
"""
import warnings

from .sqlite.database import get_DB_debug, get_DB_location, \
    initialise_database, initialise_or_create_database_at
from .sqlite.connection import path_to_dbfile

warnings.warn('The module `qcodes.dataset.database` is deprecated.\n'
              'Public features are available at the import of `qcodes`.\n'
              'Private features are available in `qcodes.dataset.sqlite.*` '
              'modules.')
