# Generate version 0 database files for qcodes' test suite to consume

import os

# NB: it's important that we do not import anything from qcodes before we
# do the git magic (which we do below), hence the relative import here
import utils as utils


fixturepath = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2])
fixturepath = os.path.join(fixturepath, 'fixtures', 'db_files')


def generate_empty_DB_file():
    """
    Generate the bare minimal DB file with no runs
    """

    import qcodes.dataset.sqlite_base as sqlite_base

    v0fixturepath = os.path.join(fixturepath, 'version0')
    os.makedirs(v0fixturepath, exist_ok=True)
    path = os.path.join(v0fixturepath, 'empty.db')

    if os.path.exists(path):
        os.remove(path)

    conn = sqlite_base.connect(path)
    sqlite_base.init_db(conn)


if __name__ == '__main__':

    gens = (generate_empty_DB_file,)

    # pylint: disable=E1101
    utils.checkout_to_old_version_and_run_generators(version=0, gens=gens)
