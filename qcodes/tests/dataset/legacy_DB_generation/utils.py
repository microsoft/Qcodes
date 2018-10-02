# General utilities for the database generation and loading scheme
from typing import Dict, List
from contextlib import contextmanager

GIT_HASHES: Dict[int, str] = {0: '78d42620fc245a975b5a615ed5e33061baac7846'}

DB_NAMES: Dict[int, List[str]] = {0: ['']}


@contextmanager
def leave_untouched(repo):

    if repo.is_dirty():
        raise ValueError('Git repository is dirty. Can not proceed.')

    current_branch = repo.active_branch
    current_commit = current_branch.commit

    yield

    repo.git.reset('--hard', current_commit)
    repo.git.checkout(current_branch)