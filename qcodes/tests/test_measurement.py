from time import sleep

import numpy as np
from unittest import TestCase
from functools import partial

from qcodes import Loop, Parameter, load_data, ParameterNode, new_job, MultiParameter

from qcodes.measurement import Measurement, Sweep, running_measurement


def verify_msmt(msmt, verification_arrays=None, allow_nan=False):
    dataset = load_data(msmt.dataset.location)

    for action_indices, data_array in msmt.data_arrays.items():
        if not allow_nan and np.any(np.isnan(data_array)):
            raise ValueError(f"Found NaN values in data array {data_array.name}")

        if verification_arrays is not None:
            verification_array = verification_arrays[action_indices]
            np.testing.assert_array_almost_equal(data_array, verification_array)

        dataset_array = dataset.arrays[data_array.array_id]

        np.testing.assert_array_almost_equal(data_array, dataset_array)

        # Test set arrays
        if not len(data_array.set_arrays) == len(dataset_array.set_arrays):
            raise RuntimeError("Unequal amount of set arrays")

        for set_array, dataset_set_array in zip(
            data_array.set_arrays, dataset_array.set_arrays
        ):
            if not allow_nan and np.any(np.isnan(set_array)):
                raise ValueError(f"Found NaN values in set array {set_array.name}")

            np.testing.assert_array_almost_equal(
                set_array.ndarray, dataset_set_array.ndarray
            )

    return dataset


class TestOldLoop(TestCase):
    def setUp(self) -> None:
        self.p_sweep = Parameter("p_sweep", set_cmd=None)
        self.p_measure = Parameter("p_measure", set_cmd=None)
        self.p_sweep.connect(self.p_measure)

    def test_old_loop_1D(self):
        loop = Loop(self.p_sweep.sweep(0, 10, 1)).each(self.p_measure, self.p_measure)
        data = loop.run(name="old_loop_1D", thread=False)

        self.assertEqual(data.metadata.get("measurement_type"), "Loop")

        # Verify that the measurement dataset records the correct measurement type
        loaded_data = load_data(data.location)
        self.assertEqual(loaded_data.metadata.get("measurement_type"), "Loop")

    def test_old_loop_2D(self):
        self.p_sweep2 = Parameter("p_sweep2", set_cmd=None)

        loop = (
            Loop(self.p_sweep.sweep(0, 5, 1))
            .loop(self.p_sweep2.sweep(0, 5, 1))
            .each(self.p_measure, self.p_measure)
        )
        loop.run(name="old_loop_2D", thread=False)

    def test_old_loop_1D_2D(self):
        self.p_sweep2 = Parameter("p_sweep2", set_cmd=None)

        loop = Loop(self.p_sweep.sweep(0, 5, 1)).each(
            self.p_measure, Loop(self.p_sweep2.sweep(0, 5, 1)).each(self.p_measure)
        )
        loop.run(name="old_loop_1D_2D", thread=False)


class TestNewLoop(TestCase):
    def setUp(self) -> None:
        self.p_sweep = Parameter("p_sweep", set_cmd=None, initial_value=10)
        self.p_measure = Parameter("p_measure", set_cmd=None)
        self.p_sweep.connect(self.p_measure, scale=10)

    def test_new_loop_1D(self):
        arrs = {}

        with Measurement("new_loop_1D") as msmt:
            for k, val in enumerate(Sweep(self.p_sweep.sweep(0, 1, 0.1))):
                arr = arrs.setdefault(
                    msmt.action_indices, np.zeros(msmt.loop_shape)
                )
                arr[k] = msmt.measure(self.p_measure)

        verify_msmt(msmt, arrs)

        # Verify that the measurement dataset records the correct measurement type
        data = load_data(msmt.dataset.location)
        self.assertEqual(data.metadata.get("measurement_type"), "Measurement")

    def test_new_loop_1D_double(self):
        arrs = {}

        with Measurement("new_loop_1D_double") as msmt:
            for k, val in enumerate(Sweep(self.p_sweep.sweep(0, 1, 0.1))):
                arr = arrs.setdefault(
                    msmt.action_indices, np.zeros(msmt.loop_shape)
                )
                arr[k] = msmt.measure(self.p_measure)

                arr = arrs.setdefault(
                    msmt.action_indices, np.zeros(msmt.loop_shape)
                )
                arr[k] = msmt.measure(self.p_measure)

        verify_msmt(msmt, arrs)

    def test_new_loop_2D(self):
        arrs = {}
        self.p_sweep2 = Parameter("p_sweep2", set_cmd=None)

        with Measurement("new_loop_1D_double") as msmt:
            for k, val in enumerate(Sweep(self.p_sweep2.sweep(0, 1, 0.1))):
                for kk, val2 in enumerate(Sweep(self.p_sweep.sweep(0, 1, 0.1))):
                    self.assertEqual(msmt.loop_shape, (11, 11))
                    arr = arrs.setdefault(
                        msmt.action_indices, np.zeros(msmt.loop_shape)
                    )
                    arr[k, kk] = msmt.measure(self.p_measure)

        verify_msmt(msmt, arrs)

    def test_new_loop_dual_sweep(self):
        with Measurement('outer') as msmt:
            self.assertEqual(msmt.action_indices, (0,))
            for _ in Sweep(range(10), 'sweep0'):
                self.assertEqual(msmt.action_indices, (0, 0))
                for _ in Sweep(range(10), 'sweep1'):
                    self.assertEqual(msmt.action_indices, (0, 0, 0))
                    msmt.measure(np.random.rand(), 'random_value1')
                self.assertEqual(msmt.action_indices, (0, 1))
                for _ in Sweep(range(10), 'sweep2'):
                    self.assertEqual(msmt.action_indices, (0, 1, 0))
                    msmt.measure(np.random.rand(), 'random_value2')

    def test_new_loop_break(self):
        arrs = {}
        self.p_sweep2 = Parameter("p_sweep2", set_cmd=None)

        with Measurement("new_loop_1D_double") as msmt:
            for k, val in enumerate(Sweep(self.p_sweep2.sweep(0, 1, 0.2))):
                for kk, val2 in enumerate(Sweep(self.p_sweep.sweep(0, 1, 0.2))):
                    self.assertEqual(msmt.loop_shape, (6, 6))
                    arr = arrs.setdefault(
                        msmt.action_indices, np.nan * np.zeros(msmt.loop_shape)
                    )
                    arr[k, kk] = msmt.measure(self.p_measure)
                    if kk == 2:
                        msmt.step_out(reduce_dimension=True)
                        break
                print('hi')

        verify_msmt(msmt, arrs, allow_nan=True)

    def test_skip_action(self):
        with Measurement('test') as msmt:
            for k in Sweep(range(5), 'sweeper'):
                msmt.measure(k, 'idx')
                if k % 2:
                    msmt.measure(2*k, 'double_idx')
                else:
                    msmt.skip()
                msmt.measure(3*k, 'triple_idx')

        arrs = {
            (0, 0): np.arange(5),
            (0, 1): [np.nan, 2, np.nan, 6, np.nan],
            (0, 2): 3 * np.arange(5)
        }

        verify_msmt(msmt, arrs, allow_nan=True)

    # def test_new_loop_0D(self):
    #     # TODO Does not work yet
    #     with Measurement('new_loop_0D') as msmt:
    #         self.assertEqual(msmt.loop_shape, ())
    #         msmt.measure(self.p_measure)

    # self.verify_msmt(msmt, arrs)


class TestNewLoopParameterNode(TestCase):
    class DictResultsNode(ParameterNode):
        def get(self):
            return {"result1": np.random.rand(), "result2": np.random.rand()}

    def test_measure_node_dict(self):
        arrs = {}
        node = self.DictResultsNode("measurable_node")
        p_sweep = Parameter("sweep", set_cmd=None)

        with Measurement("measure_node") as msmt:
            for k, val in enumerate(Sweep(p_sweep.sweep(0, 1, 0.1))):
                results = msmt.measure(node.get)

                # Save results to verification arrays
                for kk, result in enumerate(results.values()):
                    arrs.setdefault((0, 0, kk), np.zeros(msmt.loop_shape))
                    arrs[(0, 0, kk)][k] = result

    class NestedResultsNode(ParameterNode):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.results = None

        def get(self):
            self.results = {"result1": np.random.rand(), "result2": np.random.rand()}

            with Measurement(self.name) as msmt:
                for key, val in self.results.items():
                    msmt.measure(val, name=key)

            return {"wrong_result1": np.random.rand(), "wrong_result2": np.random.rand()}

    def test_measure_node_nested(self):
        arrs = {}
        node = self.NestedResultsNode("measurable_node")
        p_sweep = Parameter("sweep", set_cmd=None)

        with Measurement("measure_node") as msmt:
            for k, val in enumerate(Sweep(p_sweep.sweep(0, 1, 0.1))):
                msmt.measure(node.get)

                # Save results to verification arrays
                for kk, result in enumerate(node.results.values()):
                    arrs.setdefault((0, 0, kk), np.zeros(msmt.loop_shape))
                    arrs[(0, 0, kk)][k] = result

        verify_msmt(msmt, arrs)


class TestNewLoopFunctionResults(TestCase):
    def dict_function(self):
        return {"result1": np.random.rand(), "result2": np.random.rand()}

    def test_dict_function(self):
        arrs = {}
        p_sweep = Parameter("sweep", set_cmd=None)

        with Measurement("measure_node") as msmt:
            for k, val in enumerate(Sweep(p_sweep.sweep(0, 1, 0.1))):
                results = msmt.measure(self.dict_function)

                # Save results to verification arrays
                for kk, result in enumerate(results.values()):
                    arrs.setdefault((0, 0, kk), np.zeros(msmt.loop_shape))
                    arrs[(0, 0, kk)][k] = result

        self.assertEqual(msmt.data_groups[(0,0)].name, 'dict_function')

        verify_msmt(msmt, arrs)

    def test_dict_function_custom_name(self):
        arrs = {}
        p_sweep = Parameter("sweep", set_cmd=None)

        with Measurement("measure_node") as msmt:
            for k, val in enumerate(Sweep(p_sweep.sweep(0, 1, 0.1))):
                results = msmt.measure(self.dict_function, name='custom_name')

                # Save results to verification arrays
                for kk, result in enumerate(results.values()):
                    arrs.setdefault((0, 0, kk), np.zeros(msmt.loop_shape))
                    arrs[(0, 0, kk)][k] = result

        self.assertEqual(msmt.data_groups[(0,0)].name, 'custom_name')

        verify_msmt(msmt, arrs)

    @staticmethod
    def nested_function(results):
        with Measurement('nested_function_name') as msmt:
            for key, val in results.items():
                msmt.measure(val, name=key)

        return {"wrong_result1": np.random.rand(), "wrong_result2": np.random.rand()}

    def test_nested_function(self):
        arrs = {}
        p_sweep = Parameter("sweep", set_cmd=None)

        with Measurement("measure_node") as msmt:
            for k, val in enumerate(Sweep(p_sweep.sweep(0, 1, 0.1))):
                results = {"result1": np.random.rand(), "result2": np.random.rand()}
                msmt.measure(partial(self.nested_function, results))

                # Save results to verification arrays
                for kk, result in enumerate(results.values()):
                    arrs.setdefault((0, 0, kk), np.zeros(msmt.loop_shape))
                    arrs[(0, 0, kk)][k] = result

        self.assertEqual(msmt.data_groups[(0,0)].name, 'nested_function_name')

        verify_msmt(msmt, arrs)


class TestNewLoopArray(TestCase):
    def setUp(self) -> None:
        self.p_sweep = Parameter("p_sweep", set_cmd=None, initial_value=10)

    def test_measure_parameter_array(self):
        arrs = {}

        p_measure = Parameter("p_measure", get_cmd=lambda: np.random.rand(5))

        with Measurement("new_loop_parameter_array") as msmt:
            for k, val in enumerate(Sweep(self.p_sweep.sweep(0, 1, 0.1))):
                arr = arrs.setdefault(
                    msmt.action_indices, np.zeros(msmt.loop_shape + (5,))
                )
                result = msmt.measure(p_measure)
                arr[k] = result

        dataset = verify_msmt(msmt, arrs)

        # Perform additional test on set array
        set_array = np.broadcast_to(np.arange(5), (11, 5))
        np.testing.assert_array_almost_equal(
            dataset.arrays["p_measure_set0_0_0"], set_array
        )

    def test_measure_parameter_array_2D(self):
        arrs = {}

        p_measure = Parameter("p_measure", get_cmd=lambda: np.random.rand(5, 12))

        with Measurement("new_loop_parameter_array_2D") as msmt:
            for k, val in enumerate(Sweep(self.p_sweep.sweep(0, 1, 0.1))):
                arr = arrs.setdefault(
                    msmt.action_indices, np.zeros(msmt.loop_shape + (5, 12))
                )
                result = msmt.measure(p_measure)
                arr[k] = result

        dataset = verify_msmt(msmt, arrs)

        # Perform additional test on set arrays
        set_array = np.broadcast_to(np.arange(5), (11, 5))
        np.testing.assert_array_almost_equal(
            dataset.arrays["p_measure_set0_0_0"], set_array
        )

        set_array = np.broadcast_to(np.arange(12), (11, 5, 12))
        np.testing.assert_array_almost_equal(
            dataset.arrays["p_measure_set1_0_0_0"], set_array
        )

    class MeasurableNode(ParameterNode):
        def get(self):
            return {
                "result0D": np.random.rand(),
                "result1D": np.random.rand(5),
                "result2D": np.random.rand(5, 6),
            }

    def test_measure_parameter_array_in_node(self):
        arrs = {}

        node = self.MeasurableNode("measurable_node")

        with Measurement("new_loop_parameter_array_2D") as msmt:
            for k, val in enumerate(Sweep(self.p_sweep.sweep(0, 1, 0.1))):
                results = msmt.measure(node.get)

                # Save results to verification arrays
                for kk, result in enumerate(results.values()):
                    shape = msmt.loop_shape
                    if isinstance(result, np.ndarray):
                        shape += result.shape
                    arrs.setdefault((0, 0, kk), np.zeros(shape))
                    arrs[(0, 0, kk)][k] = result

        dataset = verify_msmt(msmt, arrs)

        # Perform additional test on set arrays
        set_array = np.broadcast_to(np.arange(5), (11, 5))
        np.testing.assert_array_almost_equal(
            dataset.arrays["result1D_set0_0_0_1"], set_array
        )

        set_array = np.broadcast_to(np.arange(5), (11, 5))
        np.testing.assert_array_almost_equal(
            dataset.arrays["result2D_set0_0_0_2"], set_array
        )
        set_array = np.broadcast_to(np.arange(6), (11, 5, 6))
        np.testing.assert_array_almost_equal(
            dataset.arrays["result2D_set1_0_0_2_0"], set_array
        )


class TestNewLoopNesting(TestCase):
    def setUp(self) -> None:
        self.p_sweep = Parameter("p_sweep", set_cmd=None, initial_value=10)
        self.p_measure = Parameter("p_measure", set_cmd=None)
        self.p_sweep.connect(self.p_measure, scale=10)

    def test_nest_measurement(self):
        def nest_measurement():
            self.assertEqual(running_measurement().action_indices, (1,))
            with Measurement("nested_measurement") as msmt:
                self.assertEqual(running_measurement().action_indices, (1, 0))
                for val in Sweep(self.p_sweep.sweep(0, 1, 0.1)):
                    self.assertEqual(running_measurement().action_indices, (1, 0, 0))
                    msmt.measure(self.p_measure)
                    self.assertEqual(running_measurement().action_indices, (1, 0, 1))
                    msmt.measure(self.p_measure)

            return msmt

        with Measurement("outer_measurement") as msmt:
            for val in Sweep(self.p_sweep.sweep(0, 1, 0.1)):
                msmt.measure(self.p_measure)
            nested_msmt = nest_measurement()

        self.assertEqual(msmt.data_groups[(1,)], nested_msmt)

    def test_double_nest_measurement(self):
        def nest_measurement():
            self.assertEqual(running_measurement().action_indices, (1,))
            with Measurement("nested_measurement") as msmt:
                self.assertEqual(running_measurement().action_indices, (1, 0))
                for val in Sweep(self.p_sweep.sweep(0, 1, 0.1)):
                    self.assertEqual(running_measurement().action_indices, (1, 0, 0))
                    with Measurement("inner_nested_measurement") as inner_msmt:
                        self.assertEqual(
                            running_measurement().action_indices, (1, 0, 0, 0)
                        )
                        for val in Sweep(self.p_sweep.sweep(0, 1, 0.1)):
                            self.assertEqual(
                                running_measurement().action_indices, (1, 0, 0, 0, 0)
                            )
                            inner_msmt.measure(self.p_measure)
                            self.assertEqual(
                                running_measurement().action_indices, (1, 0, 0, 0, 1)
                            )
                            inner_msmt.measure(self.p_measure)

            return msmt, inner_msmt

        with Measurement("outer_measurement") as msmt:
            for val in Sweep(self.p_sweep.sweep(0, 1, 0.1)):
                msmt.measure(self.p_measure)
            nested_msmt, inner_nested_msmt = nest_measurement()

        self.assertEqual(msmt.data_groups[(1,)], nested_msmt)
        self.assertEqual(msmt.data_groups[(1, 0, 0)], inner_nested_msmt)

    def test_new_loop_two_nests(self):
        with Measurement('outer') as msmt:
            self.assertEqual(msmt.action_indices, (0,))
            for _ in Sweep(range(10), 'sweep0'):
                self.assertEqual(msmt.action_indices, (0, 0))
                with Measurement('inner1') as msmt_inner:
                    self.assertEqual(msmt.action_indices, (0, 0, 0))
                    for _ in Sweep(range(10), 'sweep1'):
                        self.assertEqual(msmt.action_indices, (0, 0, 0, 0))
                        msmt.measure(np.random.rand(), 'random_value1')
                    self.assertEqual(msmt.action_indices, (0, 0, 1))
                self.assertEqual(msmt.action_indices, (0, 1))
                with Measurement('inner2') as msmt_inner:
                    self.assertEqual(msmt.action_indices, (0, 1, 0))
                    for _ in Sweep(range(10), 'sweep2'):
                        self.assertEqual(msmt.action_indices, (0, 1, 0, 0))
                        msmt.measure(np.random.rand(), 'random_value2')


class TestMeasurementThread(TestCase):
    # def setUp(self) -> None:
    #     qc.utils.threading.allow_duplicate_jobs = True
    #
    # def tearDown(self) -> None:
    #     qc.utils.threading.allow_duplicate_jobs = False

    def create_measurement(self):
        with Measurement('new_measurement') as msmt:
            msmt.pause()
            for k in Sweep([1,2,3,4], name='sweep_vals'):
                msmt.measure(123, 'test_val')

    def test_double_thread_measurement(self):
        job = new_job(self.create_measurement)
        sleep(0.2)
        with self.assertRaises(RuntimeError):
            with Measurement('new_measurement') as msmt:
                print('This line will never be reached')
                self.assertEqual(0, 1)

        running_measurement().resume()
        job.join()

    def test_double_thread_measure(self):
        job = new_job(self.create_measurement)
        sleep(0.2)
        msmt = Measurement('new_measurement')
        with self.assertRaises(RuntimeError):
            msmt.measure(123, 'test_val')

        running_measurement().resume()
        job.join()

    def test_double_thread_sweep(self):
        job = new_job(self.create_measurement)
        sleep(0.2)
        Sweep([1,2,3], 'sweep_parameter')

        running_measurement().resume()
        job.join()


class TestMultiParameter(TestCase):
    class TestMultiParameter(MultiParameter):
        def __init__(self, name):
            super().__init__(
                name=name,
                names=('val1', 'val2', 'val3'),
                units=('V', 'Hz', ''),
                shapes=((),(),())
            )

        def get_raw(self):
            return (1, 2, 3)

    def test_basic_multi_parameter(self):
        multi_parameter = self.TestMultiParameter('multi_param')
        sweep_param = Parameter('sweep_param', set_cmd=None)

        with Measurement('test_multi_parameter') as msmt:
            for val in Sweep(sweep_param.sweep(0, 10, 1)):
                msmt.measure(multi_parameter)

        self.assertEqual(msmt.data_groups[(0,0)].name, 'multi_param')

        verification_arrays = {
            (0,0,0): [1] * 11,
            (0,0,1): [2] * 11,
            (0,0,2): [3] * 11
        }
        verify_msmt(msmt, verification_arrays=verification_arrays)


class TestVerifyActions(TestCase):
    def test_simple_measurement_verification(self):
        with Measurement('test_simple_measurement_verification') as msmt:
            for val in Sweep(range(10), 'sweep_param'):
                msmt.measure(val + 2, 'msmt_param')

    def test_simple_measurement_verification_error(self):
        with self.assertRaises(RuntimeError):
            with Measurement('test_simple_measurement_verification_error') as msmt:
                for k, val in enumerate(Sweep(range(10), 'sweep_param')):
                    if k < 7:
                        msmt.measure(val + 2, 'msmt_param')
                    else:
                        msmt.measure(val + 2, 'different_msmt_param')

    def test_simple_measurement_verification_parameter(self):
        msmt_param = Parameter('msmt_param', get_cmd=np.random.rand)
        with Measurement('test_simple_measurement_verification_parameter') as msmt:
            for val in Sweep(range(10), 'sweep_param'):
                msmt.measure(msmt_param)

    def test_simple_measurement_verification_error_parameter(self):
        msmt_param = Parameter('msmt_param', get_cmd=np.random.rand)
        different_msmt_param = Parameter('different_msmt_param', get_cmd=np.random.rand)

        with self.assertRaises(RuntimeError):
            with Measurement('test_simple_measurement_verification_error_parameter') as msmt:
                for k, _ in enumerate(Sweep(range(10), 'sweep_param')):
                    if k < 7:
                        msmt.measure(msmt_param)
                    else:
                        msmt.measure(different_msmt_param)

    def test_simple_measurement_verification_no_error_parameter(self):
        msmt_param = Parameter('msmt_param', get_cmd=np.random.rand)
        msmt_param2 = Parameter('msmt_param', get_cmd=np.random.rand)
        # Notice we give the same name

        with Measurement('test_simple_measurement_verification_no_error_parameter') as msmt:
            for k, _ in enumerate(Sweep(range(10), 'sweep_param')):
                if k < 7:
                    msmt.measure(msmt_param)
                else:
                    msmt.measure(msmt_param2)

    def test_measure_callable_verification(self):
        def f():
            return {
                'param1': 1,
                'param2': 2
            }

        with Measurement('test_measure_callable_verification') as msmt:
            for k, _ in enumerate(Sweep(range(10), 'sweep_param')):
                msmt.measure(f)

    def test_measure_callable_verification_error(self):
        def f():
            return {
                'param1': 1,
                'param2': 2
            }
        def f_same():
            return {
                'param1': 1,
                'param2': 2
            }
        def different_f():
            return {
                'param2': 2,
                'param1': 1
            }

        with self.assertRaises(RuntimeError):
            with Measurement('test_measure_callable_verification_error1') as msmt:
                for k, _ in enumerate(Sweep(range(10), 'sweep_param')):
                    if k < 7:
                        msmt.measure(f)
                    else:
                        msmt.measure(different_f)

        with Measurement('test_measure_callable_verification_error2') as msmt:
            for k, _ in enumerate(Sweep(range(10), 'sweep_param')):
                if k < 7:
                    msmt.measure(f, name='f')
                else:
                    msmt.measure(f_same, name='f')

        with self.assertRaises(RuntimeError):
            with Measurement('test_measure_callable_verification_error3') as msmt:
                for k, _ in enumerate(Sweep(range(10), 'sweep_param')):
                    if k < 7:
                        msmt.measure(f, name='f')
                    else:
                        msmt.measure(different_f, name='f')

