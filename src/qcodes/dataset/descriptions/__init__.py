"""
Module containing all objects used for the description of a run.

Since run descriptions are persisted to disk, we have a strong obligation to
ensure backwards compatibility. The code is organised such that everything at
this module's top level is the CURRENT version. Older objects and the functions
necessary to convert between different versions are found in
:mod:`.versioning`.
"""
