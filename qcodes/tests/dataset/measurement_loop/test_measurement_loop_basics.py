import shutil
import pytest
import contextlib
import tempfile
from pathlib import Path

from qcodes import Parameter, ManualParameter
from qcodes.utils.dataset.doNd import LinSweep
from qcodes.dataset.measurement_loop import MeasurementLoop, Sweep
from qcodes.dataset import (
    initialise_or_create_database_at,
    load_by_run_spec,
    load_or_create_experiment,
)

# def get_data_array(dataset, label):



@pytest.fixture
def create_dummy_database():

   @contextlib.contextmanager
   def func_context_manager():
      with tempfile.TemporaryDirectory() as temporary_folder:
         temporary_folder = tempfile.TemporaryDirectory()
         print(f'Created temporary folder for database: {temporary_folder}')

         assert Path(temporary_folder.name).exists()
         db_path = Path(temporary_folder.name) / 'test_database.db'
         initialise_or_create_database_at(str(db_path))

         yield load_or_create_experiment("test_experiment")
   return func_context_manager


def test_create_measurement(create_dummy_database):
   with create_dummy_database():
      measurement_loop = MeasurementLoop('test')


def test_basic_1D_measurement(create_dummy_database):
   with create_dummy_database():
      # Initialize parameters
      p1_get = ManualParameter('p1_get')
      p1_set = ManualParameter('p1_set')

      with MeasurementLoop('test') as msmt:
         for val in Sweep(LinSweep(p1_set, 0, 1, 11)):
            assert p1_set() == val
            p1_get(val+1)
            msmt.measure(p1_get)

   


def test_basic_2D_measurement(create_dummy_database):
   with create_dummy_database():
      # Initialize parameters
      p1_get = ManualParameter('p1_get')
      p1_set = ManualParameter('p1_set')
      p2_set = ManualParameter('p2_set')

      with MeasurementLoop('test') as msmt:
         for val in Sweep(LinSweep(p1_set, 0, 1, 11)):
            assert p1_set() == val
            for val2 in Sweep(LinSweep(p2_set, 0, 1, 11)):
               assert p2_set() == val2
               p1_get(val+1)
               msmt.measure(p1_get)
   print('finished')