# Generate version 0 database files for qcodes' test suite to consume

import os

from git import Repo

# NB: it's important that we do not import anything from qcodes before we
# do the git magic (which we do below), hence the relative import here
import utils as utils

gitrepopath = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-5])
fixturepath = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2])
fixturepath = os.path.join(fixturepath, 'fixtures', 'db_files')

repo = Repo(gitrepopath)


def generate_empty_DB_file(sqlite_base):
    """
    Generate the bare minimal DB file with no runs
    """
    v0fixturepath = os.path.join(fixturepath, 'version0')
    os.makedirs(v0fixturepath, exist_ok=True)
    path = os.path.join(v0fixturepath, 'empty.db')

    if os.path.exists(path):
        os.remove(path)

    conn = sqlite_base.connect(path)
    sqlite_base.init_db(conn)


if __name__ == '__main__':

    with utils.leave_untouched(repo):  # pylint: disable=E1101

        repo.git.checkout(utils.GIT_HASHES[0])  # pylint: disable=E1101

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

        import qcodes.dataset.sqlite_base as sqlite_base

        generate_empty_DB_file(sqlite_base)
