from jsonpickle import handlers

from .jsonpickle_handler import DataSetHandler
from .data_set import DataSet

handlers.register(DataSet, DataSetHandler)