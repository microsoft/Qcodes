# General utilities for the database generation and loading scheme
from typing import Dict, List
from contextlib import contextmanager

GIT_HASHES: Dict[int, str] = {0: '78d42620fc245a975b5a615ed5e33061baac7846',
                              1: '056d59627e22fa3ca7aad4c265e9897c343f79cf'}

DB_NAMES: Dict[int, List[str]] = {0: ['']}


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
