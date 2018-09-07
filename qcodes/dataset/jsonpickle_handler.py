from jsonpickle import handlers
import jsonpickle

from qcodes.dataset.data_set import DataSet, load_by_id
from qcodes.dataset.plotting import flatten_1D_data_for_plot


class DataSetHandler(handlers.BaseHandler):

    def flatten(self, dataset: DataSet, jdict: dict) -> dict:
        jdict['run_id'] = dataset.run_id
        jdict['exp_id'] = dataset.exp_id
        jdict['result_table_name'] = dataset.table_name
        jdict['result_counter'] = dataset.number_of_results
        jdict['run_timestamp'] = dataset.run_timestamp_raw
        jdict['completed_timestamp'] = dataset.completed_timestamp_raw
        jdict['is_completed'] = dataset.completed
        jdict['parameters'] = dataset.parameters
        jdict['snapshot'] = dataset.snapshot
        #
        data = {}

        if dataset.parameters is not None:
            for parameter in dataset.parameters.split(','):
                pdata = flatten_1D_data_for_plot(dataset.get_data(parameter))
                data[parameter] = jsonpickle.encode(pdata)

        jdict['DATA'] = data

        return jdict

    # Note: the restoration is actually tricky, since we're either looking up
    # an existing run in the database or creating a new one; two very distinct
    # situations. Without a GUID, it's hard to determine which of the two
    # situations we are in. Here we just try to look up the run for a
    # proof-of-principle
    def restore(self, jdict: dict) -> DataSet:
        try:
            return load_by_id(jdict['run_id'])
        except RuntimeError:
            raise RuntimeError('No such run in database. Run insertion not '
                               'implemented yet. Can not proceed.')

