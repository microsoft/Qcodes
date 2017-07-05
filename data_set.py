#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 unga <giulioungaretti@me.com>
#
# Distributed under terms of the MIT license.

import hashlib
# import json
from typing import Any, Dict, List, Optional, Union, Sized

from param_spec import ParamSpec
from qcodes.instrument.parameter import _BaseParameter
from sqlite_base import (atomic, add_parameter, connect, create_run,
                         get_parameters, last_experiment,
                         length, modify_values,
                         modify_many_values, insert_values, insert_many_values,
                         VALUES, get_data, get_metadata)


# TODO: as of now every time a result is inserted with add_result the db is
# saved same for add_results. IS THIS THE BEHAVIOUR WE WANT?

# TODO: storing parameters in separate table

# TODO: fixix  a subset of metadata that we define well known (and create them)
# i.e. no dynamic creation of metadata columns, but add stuff to
# a json inside a 'metadata' column

def hash_from_parts(*parts: str) -> str:
    """
    Args:
        *parts:  parts to use to create hash

    Returns:
        hash created with the given parts

    """
    combined = "".join(parts)
    return hashlib.sha1(combined.encode("utf-8")).hexdigest()


# SPECS is a list of ParamSpec
SPECS = List[ParamSpec]


class CompletedError(RuntimeError):
    pass


# TODO: this clearly should be configurable
DB = "/Users/unga/Desktop/experiment.db"


class DataSet(Sized):
    def __init__(self, name, exp_id: Optional[int]= None,
                 specs: SPECS=None, values=None,
                 metadata=None) -> None:
        # TODO: handle fail here by defaulting to
        # a standard db
        self.location = DB
        self.conn = connect(self.location)
        # a standard experiment (f.ex. experiment, sample)
        if exp_id:
            self.exp_id = exp_id
        else:
            self.exp_id = last_experiment(self.conn)
        self.name = name
        id, table_name = create_run(self.conn, self.exp_id, self.name,
                                    specs, metadata)
        self.table_name = table_name
        self.id = id
        # TODO: implement this how/how ??
        if values:
            raise NotImplementedError

        self.completed = False

    def add_parameter(self, spec: ParamSpec):
        add_parameter(self.conn, self.table_name, spec)

    def get_parameters(self) -> SPECS:
        return get_parameters(self.conn, self.table_name)

    def add_parameters(self, specs: SPECS):
        add_parameter(self.conn, self.table_name, *specs)

    def add_metadata(self, tag: str, metadata: Any):
        """
        Adds metadata to the DataSet. The metadata is stored under the
        provided tag.

        Args:
            tag: represents the key in the metadata dictionary
            metadata: actual metadata

        """
        # TODO: this should follow the spec
        # option one:
        # add_meta_data(self.conn, self.id, {tag: metadata})
        # option two:
        # json_meta_data = json.dumps(metadata)
        # add_meta_data(self.conn, self.id, {"metadata": json_meta_data})
        # adding meta-data does not commit
        # self.conn.commit()
        raise NotImplemented

    def mark_complete(self) -> None:
        """Mark dataset as complete and thus read only and notify the
        subscribers"""
        self.completed = True
        # TODO: implement suscribers
        # for sub in self.subscribers.values():
        # sub.done_callback()

    def add_result(self, results: Dict[str, VALUES]) -> int:
        """
        Add a logically single result to existing parameters

        Args:
            - results: dictionary with name of a parameter as the key and the
               value to associate as the value.

        Returns:

            - index in the DataSet that the result was stored at

        If a parameter exist in the dataset and it's not in the results
        dictionary Null values are inserted.
        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this DataSet.
        It is an error to add results to a completed DataSet.
        """
        if self.completed:
            raise CompletedError
        index = insert_values(self.conn, self.table_name,
                              list(results.keys()),
                              list(results.values())
                              )
        self.conn.commit()
        return index

    def add_results(self, results: List[Dict[str, VALUES]]) -> int:
        """
        Adds a sequence of results to the DataSet.

        Args:
            - list of name, value dictionaries  where each
              dictionary provides the values for all of the parameters in
              that result.

        Returns:
            - the index in the DataSet that the **first** result was stored at

        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this DataSet.
        It is an error to add results to a completed DataSet.
        """
        keys = [list(val.keys()) for val in results]
        flattened_keys = [item for sublist in keys for item in sublist]
        values = [list(val.values()) for val in results]
        flattened_values = [item for sublist in values for item in sublist]
        len_before_add = length(self.conn, self.table_name)
        insert_many_values(self.conn, self.table_name, flattened_keys,
                           flattened_values)
        self.conn.commit()
        return len_before_add

    def modify_result(self, index: int, results: Dict[str, VALUES]) -> None:
        """ Modify a logically single result of existing parameters

        Args:
            - index: zero-based index of the result to be modified.
            - results: dictionary of updates with name of a parameter as the
               key and the value to associate as the value.


        It is an error to modify a result at an index less than zero or
        beyond the end of the DataSet.
        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this DataSet.
        It is an error to modify a result in a completed DataSet.
        """
        if self.completed:
            raise CompletedError

        modify_values(self.conn, self.table_name, index,
                      list(results.keys()),
                      list(results.values())
                      )

    def modify_results(self, start_index: int,
                       updates: List[Dict[str, VALUES]]):
        """ Modify a sequence of results in the DataSet.

        Args:
            - index: zero-based index of the result to be modified.
            - results: sequence of dictionares of updates with name of a
                parameter as the key and the value to associate as the value.


        It is an error to modify a result at an index less than zero or
        beyond the end of the DataSet.
        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this DataSet.
        It is an error to modify a result in a completed DataSet.
        """
        keys = [list(val.keys()) for val in updates]
        flattened_keys = [item for sublist in keys for item in sublist]
        values = [list(val.values()) for val in updates]
        flattened_values = [item for sublist in values for item in sublist]
        modify_many_values(self.conn,
                           self.table_name,
                           start_index,
                           flattened_keys,
                           flattened_values)

    def add_parameter_values(self, spec: ParamSpec, values: List[VALUES]):
        """
        Add a parameter to the DataSet and associates result values with the
        new parameter.

        Adds a parameter to the DataSet and associates result values with the
        new parameter.
        If the DataSet is not empty, then the count of provided
        values must equal the current count of results in the DataSet, or an
        error will result.

        It is an error to add parameters to a completed DataSet.
        # TODO: fix type cheking
        """
        # first check that the len of values (if dataset is not empty)
        # is the right size i.e. the same as the dataset
        if len(self) > 0:
            if len(values) != len(self):
                raise ValueError("Need to have {} values but got {}.".format(
                                  len(self),
                                  len(values)
                                  ))
        with atomic(self.conn):
            add_parameter(self.conn, self.table_name, spec)
            # now add values!
            results = [{spec.name: value} for value in values]
            self.add_results(results)

    def get_data(self,
                 *params: Union[str, ParamSpec, _BaseParameter],
                 start: Optional[int]= None,
                 end: Optional[int]= None)-> List[List[Any]]:
        """ Returns the values stored in the DataSet for the specified parameters.
        The values are returned as a list of parallel NumPy arrays, one array
        per parameter. The data type of each array is based on the data type
        provided when the DataSet was created. The parameter list may contain
        a mix of string parameter names, QCoDeS Parameter objects, and
        ParamSpec objects. As long as they have a `name` field. If provided,
        the start and end parameters select a range of results by result count
        (index).
        If the range is empty -- that is, if the end is less than or
        equal to the start, or if start is after the current end of the
        DataSet – then a list of empty arrays is returned.

        Args:
            - *params: string parameter names, QCoDeS Parameter objects, and
               ParamSpec objects
            - start:
            - end:

        Returns:
            - list of parallel NumPy arrays, one array per parameter
        per parameter
        """
        valid_param_names = []
        for maybeParam in params:
            if isinstance(maybeParam, str):
                valid_param_names.append(maybeParam)
                continue
            else:
                try:
                    maybeParam = maybeParam.name
                except Exception as e:
                    raise ValueError(
                        "This parameter does not have  a name") from e
            valid_param_names.append(maybeParam)
        data = get_data(self.conn, self.table_name, valid_param_names,
                        start, end)
        return data

    def get_metadata(self, tag):
        return get_metadata(self.conn, tag, self.name)

    def __len__(self):
        return length(self.conn, self.table_name)

    def __repr__(self)->str:
        out = []
        heading = f"{self.name} #{self.id}@{self.location}"
        out.append(heading)
        out.append("-" * len(heading))
        ps = self.get_parameters()
        if len(ps) > 0:
            for p in ps:
                out.append(f"{p.name} - {p.type}")

        return "\n".join(out)
