from qcodes import new_data_set, new_experiment
from qcodes.dataset.measurements import DataSaver
from qcodes.tests.dataset.test_dataset_basic import empty_temp_db, experiment

CALLBACK_COUNT = 0
CALLBACK_RUN_ID = None
CALLBACK_SNAPSHOT = None


def callback(result_list, data_set_len, state, run_id, snapshot):
    global CALLBACK_COUNT, CALLBACK_RUN_ID, CALLBACK_SNAPSHOT
    CALLBACK_COUNT += 1
    CALLBACK_RUN_ID = run_id
    CALLBACK_SNAPSHOT = snapshot

def reset_callback_globals():
    global CALLBACK_COUNT, CALLBACK_RUN_ID, CALLBACK_SNAPSHOT
    CALLBACK_COUNT = 0
    CALLBACK_RUN_ID = None
    CALLBACK_SNAPSHOT = None


def test_default_callback(experiment):
    test_set = None
    reset_callback_globals()

    try:
        DataSaver.default_callback = {
            'run_tables_subscription_callback': callback,
            'run_tables_subscription_min_wait': 1,
            'run_tables_subscription_min_count': 1,
        }
        test_set = new_data_set("test-dataset")
        test_set.add_metadata('snapshot', 123)
        DataSaver(dataset=test_set, write_period=0, parameters={})
        test_set.mark_complete()
        assert CALLBACK_SNAPSHOT == 123
        assert CALLBACK_RUN_ID > 0
        assert CALLBACK_COUNT > 0
    finally:
        DataSaver.default_callback = None
        if test_set is not None:
            test_set.conn.close()
