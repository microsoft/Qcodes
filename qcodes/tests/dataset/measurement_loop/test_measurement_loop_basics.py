import numpy as np
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
      MeasurementLoop('test')


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

   data = msmt.dataset
   assert data.name == 'test'
   assert data.parameters == 'p1_set,p1_get'

   arrays = data.get_parameter_data()
   data_arrays = arrays['p1_get']

   assert np.allclose(
      data_arrays['p1_get'],
      np.linspace(1, 2, 11)
   )
   assert np.allclose(
      data_arrays['p1_set'],
      np.linspace(0, 1, 11)
   )

   
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

   data = msmt.dataset
   assert data.name == 'test'
   assert data.parameters == 'p1_set,p2_set,p1_get'

   arrays = data.get_parameter_data()
   data_array = arrays['p1_get']['p1_get']

   assert np.allclose(
      data_array,
      np.tile(np.linspace(1, 2, 11), (11,1)).transpose()
   )

   assert np.allclose(
      arrays['p1_get']['p1_set'],
      np.tile(np.linspace(0, 1, 11), (11,1)).transpose()
   )

   assert np.allclose(
      arrays['p1_get']['p2_set'], 
      np.tile(np.linspace(0, 1, 11), (11,1))
   )
   

def test_1D_measurement_duplicate_get(create_dummy_database):
   with create_dummy_database():
      # Initialize parameters
      p1_get = ManualParameter('p1_get')
      p1_set = ManualParameter('p1_set')

      with MeasurementLoop('test') as msmt:
         for val in Sweep(LinSweep(p1_set, 0, 1, 11)):
            assert p1_set() == val
            p1_get(val+1)
            msmt.measure(p1_get)
            p1_get(val+0.5)
            msmt.measure(p1_get)
            
   data = msmt.dataset
   assert data.name == 'test'
   assert data.parameters == 'p1_set,p1_get_0,p1_get_1'
            
   arrays = data.get_parameter_data()

   offsets = {'p1_get_0': 1, 'p1_get_1': 0.5}
   for key in ['p1_get_0', 'p1_get_1']:
      data_arrays = arrays[key]

      assert np.allclose(
         data_arrays[key],
         np.linspace(0, 1, 11) + offsets[key]
      )
      assert np.allclose(
         data_arrays['p1_set'],
         np.linspace(0, 1, 11)
      )


def test_1D_measurement_duplicate_getset(create_dummy_database):
   with create_dummy_database():
      # Initialize parameters
      p1_get = ManualParameter('p1_get')
      p1_set = ManualParameter('p1_set')

      with MeasurementLoop('test') as msmt:
         for val in Sweep(LinSweep(p1_set, 0, 1, 11)):
            assert p1_set() == val
            p1_get(val+1)
            msmt.measure(p1_get)
         for val in Sweep(LinSweep(p1_set, 0, 1, 11)):
            assert p1_set() == val
            p1_get(val+0.5)
            msmt.measure(p1_get)
            
   data = msmt.dataset
   assert data.name == 'test'
   assert data.parameters == 'p1_set_0,p1_set_1,p1_get_0,p1_get_1'
            
   arrays = data.get_parameter_data()

   offsets = {'p1_get_0': 1, 'p1_get_1': 0.5}
   for k in [0, 1]:
      get_key = f'p1_get_{k}'
      set_key = f'p1_set_{k}'
      data_arrays = arrays[get_key]

      assert np.allclose(
         data_arrays[get_key],
         np.linspace(0, 1, 11) + offsets[get_key]
      )
      assert np.allclose(
         data_arrays[set_key],
         np.linspace(0, 1, 11)
      )

   
def test_2D_measurement_initialization(create_dummy_database):
   with create_dummy_database():
      # Initialize parameters
      p1_get = ManualParameter('p1_get')
      p1_set = ManualParameter('p1_set')
      p2_set = ManualParameter('p2_set')

      with MeasurementLoop('test') as msmt:
         outer_sweep = Sweep(LinSweep(p1_set, 0, 1, 11))
         for k, val in enumerate(outer_sweep):
            assert p1_set() == val
            assert outer_sweep.is_first_sweep

            inner_sweep = Sweep(LinSweep(p2_set, 0, 1, 11))
            assert not inner_sweep.is_first_sweep
            
            for val2 in inner_sweep:
               assert p2_set() == val2
               p1_get(val+1)
               msmt.measure(p1_get)

            if not k:
               assert not msmt.data_handler.initialized
            else:
               assert msmt.data_handler.initialized