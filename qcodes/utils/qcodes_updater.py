import os
import subprocess
import sys
import shutil
import datetime
import warnings

import qcodes

def qcodes_backup(env_name: str = 'qcodes', env_backup_name: str = 'qcodes_backup'):
    """
    Back up QCoDeS Anaconda Python environment.
    Args:
        env_name: Name of the environment to be updated
        env_backup_name: Name of the back up environment to be created
    """
    _source = os.sep.join(qcodes.__file__.split(os.sep)[:-2])

    _time = str(datetime.date.today()) # Current date.
    env_backup_name = env_backup_name + '_' + _time

    print(f"Existing {env_name} environment will be backed up as {env_backup_name}...\n")

    subprocess.run(f'conda create --name {env_backup_name} --clone {env_name}', shell=True)

    # Copy QCoDeS root to qcodes_backup

    _destination = _source + '_backup' + '_' + _time

    shutil.copytree(_source, _destination, symlinks=True, ignore=None)

    # Re-install QCoDeS from the back up root

    subprocess.run(f'activate {env_backup_name} && pip uninstall qcodes && pip install -e {_destination}', shell=True)

    return _source, _destination, env_name, env_backup_name

def update_qcodes_installation(conda_update: bool = True, env_update: bool = True, qcodes_update: bool = True):

    """
    Update current QCoDeS Anaconda Python installation.
    Args:
        conda_update: Update 'conda' manager of Anaconda Python installation
        env_update: Update QCoDeS python environment
        qcodes_update: Updates QCoDeS module
    Raises:
        ImportError: If QCoDeS module cannot be imported after the update
        CalledProcessError: If working git tree is not clean
    """
    # Will update the Conda environment.
    if conda_update:
        print("Updating Conda...\n")
        subprocess.run('conda update -n base conda -c defaults', shell=True)

    #Back up QCoDeS environment before any relevant update.
    if (env_update or qcodes_update):
        _source, _destination, env_name, env_backup_name = qcodes_backup()

    # Update QCoDeS Environment

    # Note that Conda will not upgrade a package, even if there is a newer
    # version, in the case that the package satisfies the requirement specified
    # by QCoDeS.

    if env_update:
        print("Updating QCoDeS environment...\n")
        subprocess.run('conda env update', shell=True, cwd=_source)

    # Update QCoDeS

    if qcodes_update:
        print('Making a pull request from QCoDeS repositories...\n')
        try:
            subprocess.run('git checkout master && git pull', shell=True, cwd=_source, check=True)
            print("Updating QCoDeS...\n")
            subprocess.run(f'pip install -e {_source}', shell=True)
            print(f"QCoDeS version {qcodes.__version__} is succesfully installed.")
            sys.exit()
        except subprocess.CalledProcessError:
            print("Please clean your git tree and try again.")
            sys.exit()
        except ImportError:
            warnings.warn("An unknown issue occured during update.\nThe changes shall be rolled back.", UserWarning, 2)

            subprocess.run('conda deactivate && conda remove --name qcodes --all', shell=True)

            print("Cloning QCoDeS from back up...\n")

            subprocess.run(f'conda create --name {env_name} --clone {env_backup_name}', shell=True)
            subprocess.run(f'conda remove --name {env_backup_name} --all', shell=True)

            # Roll back to the roots

            shutil.rmtree(_source)
            shutil.copytree(_destination, _source, symlinks=True, ignore=None)

            subprocess.run(f'activate qcodes && pip uninstall qcodes && pip install -e {_source}', shell=True)

            shutil.rmtree(_destination)

            sys.exit()
