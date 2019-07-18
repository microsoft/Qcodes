import os
import subprocess
import inspect
import shutil
import datetime
import warnings
import git

import qcodes

def qcodes_backup(env_name: str = 'qcodes', env_backup_name: str = 'qcodes_backup'):
    """
    Back up QCoDeS Anaconda Python environment.

    Args:
        env_name: Name of the environment to be updated
        env_backup_name: Name of the back up environment to be created

    Raises:
        git.exc.GitError: If working tree is dirty
    """
    _source = os.sep.join(qcodes.__file__.split(os.sep)[:-2])

    # Make sure that git working tree is clean
    repo = git.Repo(_source)
    if (inspect.stack()[1].function == 'update_qcodes_installation' and repo.is_dirty()):
        raise git.exc.GitError
    elif (inspect.stack()[1].function == 'conda_env_update' and repo.is_dirty()):
        raise git.exc.GitError
    elif repo.is_dirty():
        print("QCoDeS working tree is not clean. Please commit or stash your changes, and try again.")
    else:
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

def conda_env_update(conda_update: bool = True, env_update: bool = True):
    """
    Update Conda package manager and QCoDeS Anaconda Python environment

    Args:
        conda_update: Update Conda pakage manager
        env_update: Update Anaconda Python environment

    Raises:
        git.exc.GitError: If working tree is dirty
    """
    def execute_update(_source: str):
        # Pull QCoDeS master
        repo = git.Repo(_source)
        _git = repo.git
        _git.checkout('master')
        _git.pull()

        # Update Conda
        if conda_update:
            print("Updating Conda...\n")
            subprocess.run('conda update -n base conda -c defaults', shell=True)

        # Update QCoDeS Anaconda Python environment
        # Note that Conda will not upgrade a package, even if there is a newer
        # version, in the case that the package satisfies the requirement specified
        # by QCoDeS.
        if env_update:
            print("Updating QCoDeS environment...\n")
            subprocess.run('conda env update', shell=True, cwd=_source)

    try:
        if not inspect.stack()[1].function == 'update_qcodes_installation':
            # Back up first
            _source, _destination, env_name, env_backup_name = qcodes_backup()
            # Then, update
            execute_update(_source)
        else:
            execute_update(_source)
    except git.exc.GitError:
        print("QCoDeS working tree is not clean. Please commit or stash your changes, and try again.")

def update_qcodes_installation():

    """
    Update current QCoDeS Anaconda Python installation.

    Args:
     qcodes_update: Updates QCoDeS module

    Raises:
        ImportError: If QCoDeS module cannot be imported after the update
        git.exc.GitError: If working tree is dirty
    """
    try:
        # Backup first
        _source, _destination, env_name, env_backup_name = qcodes_backup()
        # Now update environment
        conda_env_update()
        # Update QCoDeS
        print('Updating QCoDeS from master...\n')
        subprocess.run(f'pip install -e {_source}', shell=True)
        print(f"QCoDeS version {qcodes.__version__} is succesfully installed.")
    except git.exc.GitError:
        print("QCoDeS working tree is not clean. Please commit or stash your changes, and try again.")
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