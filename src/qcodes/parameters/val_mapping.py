from __future__ import annotations

from collections import OrderedDict
from typing import TypeVar

T = TypeVar("T")


def create_on_off_val_mapping(
    on_val: T | bool = True, off_val: T | bool = False
) -> OrderedDict[str | bool, T | bool]:
    """
    Returns a value mapping which maps inputs which reasonably mean "on"/"off"
    to the specified ``on_val``/``off_val`` which are to be sent to the
    instrument. This value mapping is such that, when inverted,
    ``on_val``/``off_val`` are mapped to boolean ``True``/``False``.
    """
    # Here are the lists of inputs which "reasonably" mean the same as
    # "on"/"off" (note that True/False values will be added below, and they
    # will always be added)
    ons_: tuple[str | bool, ...] = ("On", "ON", "on", "1")
    offs_: tuple[str | bool, ...] = ("Off", "OFF", "off", "0")

    # The True/False values are added at the end of on/off inputs,
    # so that after inversion True/False will be the only remaining
    # keys in the inverted value mapping dictionary.
    # NOTE that using 1/0 integer values will also work implicitly
    # due to `hash(True) == hash(1)`/`hash(False) == hash(0)`,
    # hence there is no need for adding 1/0 values explicitly to
    # the list of `ons` and `offs` values.
    ons = ons_ + (True,)
    offs = offs_ + (False,)
    all_vals_tuples = [(on, on_val) for on in ons] + [(off, off_val) for off in offs]
    return OrderedDict(all_vals_tuples)
