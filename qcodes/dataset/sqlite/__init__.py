r"""
This module encapsulates QCoDeS database: its schema, structure, convenient
and relevant queries, wrapping around :mod:`sqlite3`, etc.

The dependency structure of the sub-modules is the following:

::

         .connection     .settings
           /   |   \         |
          /    |    \        |
         /     |     V       V
        |      |  .query_helpers
        |      |   |       |
        |      V   V       |
        |  .db_upgrades    |
        |      /           V
        |     /        .queries
        v    v
      .database

"""
