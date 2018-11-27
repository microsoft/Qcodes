# Generate version 3 database files for qcodes' test suite to consume

import os
import numpy as np

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

    v3fixturepath = os.path.join(fixturepath, 'version3')
    os.makedirs(v3fixturepath, exist_ok=True)
    path = os.path.join(v3fixturepath, 'empty.db')

    if os.path.exists(path):
        os.remove(path)

    sqlite_base.connect(path)


def generate_DB_file_with_some_runs_having_not_run_descriptions():
    """
    Generate a .db-file with a handful of runs some of which lack run
    description or have it as empty object (based on a real case).

    Generated runs:
        #1: run with parameters and correct run description
        #2: run with parameters but run description is NULL
        #3: run with parameters but run description is empty RunDescriber
        #4: run without parameters but run description is NULL
    """
    v3fixturepath = os.path.join(fixturepath, 'version3')
    os.makedirs(v3fixturepath, exist_ok=True)
    path = os.path.join(v3fixturepath, 'some_runs_without_run_description.db')

    if os.path.exists(path):
        os.remove(path)

    from qcodes.dataset.measurements import Measurement
    from qcodes.dataset.experiment_container import Experiment
    from qcodes import Parameter
    from qcodes.dataset.descriptions import RunDescriber
    from qcodes.dataset.dependencies import InterDependencies

    exp = Experiment(path_to_db=path,
                     name='experiment_1',
                     sample_name='no_sample_1')
    conn = exp.conn

    # Now make some parameters to use in measurements
    params = []
    for n in range(5):
        params.append(Parameter(f'p{n}', label=f'Parameter {n}',
                                unit=f'unit {n}', set_cmd=None, get_cmd=None))

    # Set up a measurement

    meas = Measurement(exp)
    meas.register_parameter(params[0])
    meas.register_parameter(params[1])
    meas.register_parameter(params[2], basis=(params[0],))
    meas.register_parameter(params[3], basis=(params[1],))
    meas.register_parameter(params[4], setpoints=(params[2], params[3]))

    # Initially make 3 correct runs

    run_ids = []

    for _ in range(3):

        with meas.run() as datasaver:

            for x in np.random.rand(10):
                for y in np.random.rand(10):
                    z = np.random.rand()
                    datasaver.add_result((params[2], x),
                                         (params[3], y),
                                         (params[4], z))

        run_ids.append(datasaver.run_id)

    assert [1, 2, 3] == run_ids, 'Run ids of generated runs are not as ' \
                                 'expected after generating runs #1-3'

    # Formulate SQL query for adjusting run_description column

    set_run_description_sql = f"""
               UPDATE runs
               SET run_description = ?
               WHERE run_id == ?
               """

    # Make run_description of run #2 NULL

    conn.execute(set_run_description_sql, (None, run_ids[1]))
    conn.commit()  # just to be sure

    # Make run_description of run #3 equivalent to an empty RunDescriber

    empty_run_description = RunDescriber(InterDependencies()).to_json()
    conn.execute(set_run_description_sql, (empty_run_description, run_ids[2]))
    conn.commit()  # just to be sure

    # Set up a measurement without parameters, and create run #4 out of it

    meas_no_params = Measurement(exp)

    with meas_no_params.run() as datasaver:
        pass

    run_ids.append(datasaver.run_id)

    assert [1, 2, 3, 4] == run_ids, 'Run ids of generated runs are not as ' \
                                    'expected after generating run #4'

    # Make run_description of run #4 NULL

    conn.execute(set_run_description_sql, (None, run_ids[3]))
    conn.commit()  # just to be sure


if __name__ == '__main__':

    gens = (generate_empty_DB_file,
            generate_DB_file_with_some_runs_having_not_run_descriptions)

    # pylint: disable=E1101
    utils.checkout_to_old_version_and_run_generators(version=3, gens=gens)
