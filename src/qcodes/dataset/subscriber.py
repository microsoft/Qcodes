from __future__ import annotations

import functools
import logging
import time
from queue import Empty, Queue
from threading import Thread
from typing import TYPE_CHECKING, Any

from qcodes.dataset.sqlite.connection import atomic_transaction

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from qcodes.dataset.data_set import DataSet


class _Subscriber(Thread):
    """
    Class to add a subscriber to a :class:`.DataSet`. The subscriber gets called every
    time an insert is made to the results_table.

    The _Subscriber is not meant to be instantiated directly, but rather used
    via the 'subscribe' method of the :class:`.DataSet`.

    NOTE: A subscriber should be added *after* all parameters have been added.

    NOTE: Special care shall be taken when using the *state* object: it is the
    user's responsibility to operate with it in a thread-safe way.
    """

    def __init__(
        self,
        dataSet: DataSet,
        id_: str,
        callback: Callable[..., None],
        state: Any | None = None,
        loop_sleep_time: int = 0,  # in milliseconds
        min_queue_length: int = 1,
        callback_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self._id = id_

        self.dataSet = dataSet
        self.table_name = dataSet.table_name
        self._data_set_len = len(dataSet)

        self.state = state

        self.data_queue: Queue[Any] = Queue()
        self._queue_length: int = 0
        self._stop_signal: bool = False
        # convert milliseconds to seconds
        self._loop_sleep_time = loop_sleep_time / 1000
        self.min_queue_length = min_queue_length

        if callback_kwargs is None or len(callback_kwargs) == 0:
            self.callback = callback
        else:
            self.callback = functools.partial(callback, **callback_kwargs)

        self.callback_id = f"callback{self._id}"
        self.trigger_id = f"sub{self._id}"

        conn = dataSet.conn

        conn.create_function(self.callback_id, -1, self._cache_data_to_queue)

        parameters = dataSet.get_parameters()
        sql_param_list = ",".join(f"NEW.{p.name}" for p in parameters)
        sql_create_trigger_for_callback = f"""
        CREATE TRIGGER {self.trigger_id}
            AFTER INSERT ON '{self.table_name}'
        BEGIN
            SELECT {self.callback_id}({sql_param_list});
        END;"""
        atomic_transaction(conn, sql_create_trigger_for_callback)

        self.log = logging.getLogger(f"_Subscriber {self._id}")

    def _cache_data_to_queue(self, *args: Any) -> None:
        self.data_queue.put(args)
        self._data_set_len += 1
        self._queue_length += 1

    def run(self) -> None:
        self.log.debug("Starting subscriber")
        self._loop()

    @staticmethod
    def _exhaust_queue(queue: Queue[Any]) -> list[Any]:
        result_list = []
        while True:
            try:
                result_list.append(queue.get(block=False))
            except Empty:
                break
        return result_list

    def _call_callback_on_queue_data(self) -> None:
        result_list = self._exhaust_queue(self.data_queue)
        self.callback(result_list, self._data_set_len, self.state)

    def _loop(self) -> None:
        while True:
            if self._stop_signal:
                self._clean_up()
                break

            if self._queue_length >= self.min_queue_length:
                self._call_callback_on_queue_data()
                self._queue_length = 0

            time.sleep(self._loop_sleep_time)

            if self.dataSet.completed:
                self._call_callback_on_queue_data()
                break

    def done_callback(self) -> None:
        self._call_callback_on_queue_data()

    def schedule_stop(self) -> None:
        if not self._stop_signal:
            self.log.debug("Scheduling stop")
            self._stop_signal = True

    def _clean_up(self) -> None:
        self.log.debug("Stopped subscriber")
