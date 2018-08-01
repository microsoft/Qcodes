from typing import List, Union, cast
from itertools import combinations

from qcodes.dataset.data_set import DataSet
from qcodes.dataset.data_set import new_data_set, load_by_id


def _params_are_the_same(ds1: DataSet, ds2: DataSet) -> bool:
    params1 = sorted(ds1.get_parameters(), key=lambda x: x.name)
    params2 = sorted(ds2.get_parameters(), key=lambda x: x.name)
    return params1 == params2


def merge(datasets: List[Union[int, DataSet]]) -> DataSet:
    """
    Merge two or more datasets together into a new dataset containing all
    the data of the original ones. The merge can only happen if the original
    datasets have identical parameters.

    Args:
        datasets: A sequence of datasets to merge. Can either be DataSet
        objects or run_ids.
    """

    # Step 0: convert any input run_ids to DataSets
    for ds in datasets:
        if isinstance(ds, int):
            datasets.insert(datasets.index(ds), load_by_id(ds))
            datasets.remove(ds)

    dsets = cast(List[DataSet], datasets)

    # Step 1: verify that all parameters match
    ds_pairs = combinations(dsets, 2)
    for pair in ds_pairs:
        if not _params_are_the_same(pair[0], pair[1]):
            raise ValueError(f'Can not merge datasets {pair[0]} and {pair[1]},'
                             ' parameters are not the same in those datasets.')

    # Step 2: construct a new dataset
    new_ds_name = f'merge_of_{"_".join([str(ds.run_id) for ds in dsets])}'
    new_ds = new_data_set(name=new_ds_name,
                          specs=dsets[0].get_parameters())

    # Step 3: fill in the data of the old datasets
    results = []
    for ds in dsets:
        params = ds.get_parameters()
        data = ds.get_data(*params)
        for data_point in data:
            results.append({p.name: val for p, val in zip(params, data_point)})
    new_ds.add_results(results)

    return new_ds
