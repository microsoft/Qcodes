# General utilities for the database generation and loading scheme
from typing import Dict, List, Tuple
from contextlib import contextmanager
import os

from git import Repo

# A brief overview of what each version introduces:
#
# Version 0: the original table schema, runs, experiments, layouts,
# dependencies, result-tables
#
# Version 1: a GUID column is added to the runs table
#
# Version 2: indices are added to runs; GUID and exp_id
#


GIT_HASHES: Dict[int, str] = {0: '78d42620fc245a975b5a615ed5e33061baac7846',
                              1: '056d59627e22fa3ca7aad4c265e9897c343f79cf',
                              2: '5202255924542dad6841dfe3d941a7f80c43956c'}

gitrepopath = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-5])
repo = Repo(gitrepopath)


@contextmanager
def leave_untouched(repo):
    """
    Leave a git repository untouched by whatever fiddling around we need to do
    We support both the case of the repo being initially in a detached head
    state (relevant for Travis) and the -hopefully- normal case for users of
    being at the tip of a branch
    """

    if repo.is_dirty():
        raise ValueError('Git repository is dirty. Can not proceed.')

    was_detached = repo.head.is_detached

    if not was_detached:
        current_branch = repo.active_branch
    current_commit = repo.head.commit

    try:
        yield

    finally:
        repo.git.reset('--hard', current_commit)
        if not was_detached:
            repo.git.checkout(current_branch)


def checkout_to_old_version_and_run_generators(version: int,
                                               gens: Tuple) -> None:
    """
    Check out the repo to an older version and run the generating functions
    supplied.
    """

    with leave_untouched(repo):

        repo.git.checkout(GIT_HASHES[version])

        # If QCoDeS is not installed in editable mode, it makes no difference
        # to do our git magic, since the import will be from site-packages in
        # the environment folder, and not from the git-managed folder
        import qcodes
        qcpath = os.sep.join(qcodes.__file__.split(os.sep)[:-2])

        # Windows and paths... There can be random un-capitalizations
        if qcpath.lower() != gitrepopath.lower():
            raise ValueError('QCoDeS does not seem to be installed in editable'
                             ' mode, can not proceed. To use this script, '
                             'uninstall QCoDeS and reinstall it with pip '
                             'install -e <path-to-qcodes-folder>')

        for generator in gens:
            generator()
