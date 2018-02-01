Benchmark results 
=================
.. image:: TimeSuiteAddResultContext.png
	:width: 1000px
	:align: center
	:height: 800px

.. code-block:: python

    def time_range(self, insertion_size):
        self._x(0)
        self._m.get = lambda: np.arange(insertion_size)

        with self._meas.run() as datasaver:
            datasaver.add_result((self._x, self._x()), (self._m, self._m()))
.. image:: TimeSuiteAddResults.png
	:width: 1000px
	:align: center
	:height: 800px

.. code-block:: python

    def time_range(self, insertion_size):
        t_values = np.linspace(-1, 1, insertion_size)
        results = [{"t": t, "x": 2 * t ** 2 + 1} for t in t_values]
        self._data_set.add_results(results)
.. image:: TimeSuiteAddResult.png
	:width: 1000px
	:align: center
	:height: 800px

.. code-block:: python

    def time_range(self, insertion_size):

        t_values = np.linspace(-1, 1, insertion_size)
        results = [{"t": t, "x": 2 * t ** 2 + 1} for t in t_values]

        for result in results:
            self._data_set.add_result(result)
.. image:: TimeSuiteAddArrayResults.png
	:width: 1000px
	:align: center
	:height: 800px

.. code-block:: python

    def time_range(self, insertion_size):
        t_values = np.linspace(-1, 1, insertion_size)
        results = [{"t": t, "x": np.array([2 * t**2 + 1, t**3 - 1])} for t in t_values]
        self._data_set.add_results(results)
