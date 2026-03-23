"""
Extended tests for the validators module to improve coverage.

Covers edge cases and helper functions not fully exercised by existing tests:
validate_all, range_str, Validator base class, Nothing, Bool, Strings, Enum,
OnOff, Multiples, PermissiveMultiples, MultiType, MultiTypeOr, MultiTypeAnd,
Arrays, Lists, Sequence, Callable, Dict.
"""

from __future__ import annotations

import numpy as np
import pytest

from qcodes.validators import (
    Anything,
    Arrays,
    Bool,
    Dict,
    Enum,
    Ints,
    Lists,
    Multiples,
    MultiType,
    MultiTypeAnd,
    MultiTypeOr,
    Nothing,
    Numbers,
    OnOff,
    PermissiveMultiples,
    Sequence,
    Strings,
    Validator,
)
from qcodes.validators import Callable as CallableValidator
from qcodes.validators.validators import range_str, validate_all


# ---------------------------------------------------------------------------
# validate_all
# ---------------------------------------------------------------------------
class TestValidateAll:
    def test_multiple_valid(self) -> None:
        validate_all(
            (Numbers(), 1),
            (Strings(), "hello"),
            (Ints(), 42),
        )

    def test_invalid_value_raises(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            validate_all(
                (Numbers(), 1),
                (Strings(), 123),  # not a string
            )

    def test_context_appears_in_error(self) -> None:
        with pytest.raises((TypeError, ValueError), match="my context"):
            validate_all(
                (Strings(), 999),
                context="my context",
            )

    def test_context_with_argument_index(self) -> None:
        with pytest.raises((TypeError, ValueError), match="argument 0"):
            validate_all(
                (Ints(), "not_int"),
                context="ctx",
            )

    def test_empty_args(self) -> None:
        validate_all()  # no validators -> no error


# ---------------------------------------------------------------------------
# range_str
# ---------------------------------------------------------------------------
class TestRangeStr:
    def test_both_set_equal(self) -> None:
        assert range_str(5, 5, "x") == " x=5"

    def test_both_set_different(self) -> None:
        assert range_str(1, 10, "v") == " 1<=v<=10"

    def test_only_max(self) -> None:
        assert range_str(None, 10, "v") == " v<=10"

    def test_only_min(self) -> None:
        assert range_str(5, None, "v") == " v>=5"

    def test_neither(self) -> None:
        assert range_str(None, None, "v") == ""


# ---------------------------------------------------------------------------
# Validator base class
# ---------------------------------------------------------------------------
class TestValidatorBase:
    def test_validate_raises_not_implemented(self) -> None:
        v = Validator()
        with pytest.raises(NotImplementedError):
            v.validate(42)

    def test_valid_values_empty_by_default(self) -> None:
        v = Validator()
        assert v.valid_values == ()

    def test_is_numeric_false_by_default(self) -> None:
        assert Validator.is_numeric is False


# ---------------------------------------------------------------------------
# Nothing
# ---------------------------------------------------------------------------
class TestNothing:
    def test_validate_raises_runtime_error(self) -> None:
        n = Nothing("disabled")
        with pytest.raises(RuntimeError, match="disabled"):
            n.validate(42)

    def test_validate_includes_context(self) -> None:
        n = Nothing("broken")
        with pytest.raises(RuntimeError, match="my_ctx"):
            n.validate(0, context="my_ctx")

    def test_repr(self) -> None:
        n = Nothing("some reason")
        assert repr(n) == "<Nothing(some reason)>"

    def test_reason_getter(self) -> None:
        n = Nothing("test")
        assert n.reason == "test"

    def test_reason_setter(self) -> None:
        n = Nothing("old")
        n.reason = "new"
        assert n.reason == "new"

    def test_empty_reason_defaults(self) -> None:
        n = Nothing("")
        assert n.reason == "Nothing Validator"


# ---------------------------------------------------------------------------
# Bool
# ---------------------------------------------------------------------------
class TestBoolExtended:
    def test_np_bool_accepted(self) -> None:
        b = Bool()
        b.validate(np.bool_(True))
        b.validate(np.bool_(False))

    def test_non_bool_raises(self) -> None:
        b = Bool()
        with pytest.raises(TypeError, match="not Boolean"):
            b.validate(1)

    def test_repr(self) -> None:
        assert repr(Bool()) == "<Boolean>"

    def test_valid_values(self) -> None:
        assert Bool().valid_values == (True, False)


# ---------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------
class TestStringsExtended:
    def test_min_length_boundary(self) -> None:
        s = Strings(min_length=3)
        s.validate("abc")  # exactly min_length
        with pytest.raises(ValueError):
            s.validate("ab")

    def test_max_length_boundary(self) -> None:
        s = Strings(max_length=5)
        s.validate("abcde")  # exactly max_length
        with pytest.raises(ValueError):
            s.validate("abcdef")

    def test_invalid_min_length_type(self) -> None:
        with pytest.raises(
            TypeError, match="min_length must be a non-negative integer"
        ):
            Strings(min_length=-1)

    def test_invalid_min_length_float(self) -> None:
        with pytest.raises(
            TypeError, match="min_length must be a non-negative integer"
        ):
            Strings(min_length=1.5)  # type: ignore[arg-type]

    def test_invalid_max_length(self) -> None:
        with pytest.raises(TypeError, match="max_length must be a positive integer"):
            Strings(max_length=0)

    def test_max_less_than_min(self) -> None:
        with pytest.raises(TypeError, match="max_length must be a positive integer"):
            Strings(min_length=5, max_length=3)

    def test_non_string_raises(self) -> None:
        s = Strings()
        with pytest.raises(TypeError, match="not a string"):
            s.validate(42)

    def test_repr_with_constraints(self) -> None:
        s = Strings(min_length=2, max_length=10)
        r = repr(s)
        assert "len" in r
        assert "Strings" in r

    def test_repr_without_constraints(self) -> None:
        s = Strings()
        assert repr(s) == "<Strings>"

    def test_properties(self) -> None:
        s = Strings(min_length=3, max_length=50)
        assert s.min_length == 3
        assert s.max_length == 50

    def test_valid_values_with_min_length(self) -> None:
        s = Strings(min_length=3)
        # valid_values should be a string of length min_length
        assert len(s.valid_values[0]) == 3


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------
class TestEnumExtended:
    def test_unhashable_raises_type_error(self) -> None:
        e = Enum("a", "b")
        with pytest.raises(TypeError):
            e.validate([1, 2])  # list is unhashable

    def test_unhashable_error_includes_context(self) -> None:
        e = Enum("a", "b")
        with pytest.raises(TypeError, match="test_ctx"):
            e.validate([1, 2], context="test_ctx")

    def test_values_returns_copy(self) -> None:
        e = Enum("x", "y")
        vals = e.values
        vals.add("z")
        assert "z" not in e.values

    def test_repr_format(self) -> None:
        e = Enum("a")
        r = repr(e)
        assert r.startswith("<Enum:")
        assert r.endswith(">")

    def test_no_values_raises(self) -> None:
        with pytest.raises(TypeError, match="at least one value"):
            Enum()


# ---------------------------------------------------------------------------
# OnOff
# ---------------------------------------------------------------------------
class TestOnOff:
    def test_on_valid(self) -> None:
        OnOff().validate("on")

    def test_off_valid(self) -> None:
        OnOff().validate("off")

    def test_other_string_rejected(self) -> None:
        with pytest.raises(ValueError):
            OnOff().validate("yes")

    def test_non_string_rejected(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            OnOff().validate(1)

    def test_valid_values(self) -> None:
        assert set(OnOff().valid_values) == {"on", "off"}


# ---------------------------------------------------------------------------
# Multiples
# ---------------------------------------------------------------------------
class TestMultiplesExtended:
    def test_zero_divisor_raises(self) -> None:
        with pytest.raises(TypeError, match="positive integer"):
            Multiples(divisor=0)

    def test_negative_divisor_raises(self) -> None:
        with pytest.raises(TypeError, match="positive integer"):
            Multiples(divisor=-3)

    def test_float_divisor_raises(self) -> None:
        with pytest.raises(TypeError, match="positive integer"):
            Multiples(divisor=2.5)  # type: ignore[arg-type]

    def test_non_multiple_raises(self) -> None:
        m = Multiples(divisor=3)
        with pytest.raises(ValueError, match="not a multiple"):
            m.validate(7)

    def test_valid_multiple(self) -> None:
        m = Multiples(divisor=5)
        m.validate(15)
        m.validate(0)

    def test_repr_includes_divisor(self) -> None:
        m = Multiples(divisor=4)
        assert "Multiples of 4" in repr(m)

    def test_divisor_property(self) -> None:
        m = Multiples(divisor=7)
        assert m.divisor == 7


# ---------------------------------------------------------------------------
# PermissiveMultiples
# ---------------------------------------------------------------------------
class TestPermissiveMultiplesExtended:
    def test_int_divisor_int_value(self) -> None:
        """With int divisor and int value, uses Multiples path."""
        pm = PermissiveMultiples(divisor=3)
        pm.validate(9)
        pm.validate(-6)

    def test_float_divisor_float_value(self) -> None:
        pm = PermissiveMultiples(divisor=0.1)
        pm.validate(0.3)

    def test_almost_multiple_within_precision(self) -> None:
        pm = PermissiveMultiples(divisor=0.1, precision=1e-6)
        pm.validate(0.30000000001)  # within precision

    def test_not_multiple_raises(self) -> None:
        pm = PermissiveMultiples(divisor=0.1, precision=1e-9)
        with pytest.raises(ValueError, match="not a multiple"):
            pm.validate(0.15)

    def test_zero_divisor_raises(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            PermissiveMultiples(divisor=0)

    def test_zero_value_always_passes(self) -> None:
        pm = PermissiveMultiples(divisor=7)
        pm.validate(0)

    def test_repr(self) -> None:
        pm = PermissiveMultiples(divisor=5)
        r = repr(pm)
        assert "PermissiveMultiples" in r
        assert "5" in r

    def test_divisor_property(self) -> None:
        pm = PermissiveMultiples(divisor=3)
        assert pm.divisor == 3

    def test_precision_property(self) -> None:
        pm = PermissiveMultiples(divisor=2, precision=1e-6)
        assert pm.precision == 1e-6

    def test_precision_setter(self) -> None:
        pm = PermissiveMultiples(divisor=2)
        pm.precision = 1e-3
        assert pm.precision == 1e-3

    def test_divisor_setter(self) -> None:
        pm = PermissiveMultiples(divisor=2)
        pm.divisor = 5
        assert pm.divisor == 5

    def test_divisor_setter_zero_raises(self) -> None:
        pm = PermissiveMultiples(divisor=2)
        with pytest.raises(ValueError, match="zero"):
            pm.divisor = 0

    def test_is_numeric(self) -> None:
        assert PermissiveMultiples(divisor=2).is_numeric is True

    def test_int_divisor_non_multiple_int(self) -> None:
        """Int divisor + int value that is not a multiple -> error from Multiples."""
        pm = PermissiveMultiples(divisor=3)
        with pytest.raises(ValueError):
            pm.validate(7)

    def test_float_divisor_sets_mulval_none(self) -> None:
        """Float divisor should not create a Multiples sub-validator."""
        pm = PermissiveMultiples(divisor=0.5)
        assert pm._mulval is None


# ---------------------------------------------------------------------------
# MultiType
# ---------------------------------------------------------------------------
class TestMultiTypeExtended:
    def test_or_all_pass(self) -> None:
        mt = MultiType(Numbers(), Strings())
        mt.validate(42)
        mt.validate("hello")

    def test_or_none_pass(self) -> None:
        mt = MultiType(Numbers(), Strings())
        with pytest.raises(ValueError):
            mt.validate([1, 2])

    def test_and_all_pass(self) -> None:
        mt = MultiType(Numbers(min_value=0), Numbers(max_value=10), combiner="AND")
        mt.validate(5)

    def test_and_one_fails(self) -> None:
        mt = MultiType(Numbers(min_value=0), Numbers(max_value=10), combiner="AND")
        with pytest.raises(ValueError):
            mt.validate(20)

    def test_no_validators_raises(self) -> None:
        with pytest.raises(TypeError, match="at least one Validator"):
            MultiType()

    def test_non_validator_arg_raises(self) -> None:
        with pytest.raises(TypeError, match="each argument must be a Validator"):
            MultiType("not_a_validator")  # type: ignore[arg-type]

    def test_invalid_combiner_raises(self) -> None:
        with pytest.raises(TypeError, match="combiner"):
            MultiType(Numbers(), combiner="XOR")  # type: ignore[arg-type]

    def test_is_numeric_with_numeric_sub(self) -> None:
        mt = MultiType(Numbers(), Strings())
        assert mt.is_numeric is True

    def test_is_numeric_without_numeric_sub(self) -> None:
        mt = MultiType(Strings(), Bool())
        assert not mt.is_numeric

    def test_repr_format(self) -> None:
        mt = MultiType(Numbers(), Strings())
        r = repr(mt)
        assert r.startswith("<MultiType:")
        assert r.endswith(">")

    def test_combiner_property(self) -> None:
        mt = MultiType(Numbers(), combiner="AND")
        assert mt.combiner == "AND"

    def test_validators_property(self) -> None:
        n = Numbers()
        s = Strings()
        mt = MultiType(n, s)
        assert mt.validators == (n, s)

    def test_valid_values_combined(self) -> None:
        mt = MultiType(Numbers(min_value=0, max_value=10), Ints(min_value=0))
        assert len(mt.valid_values) > 0


# ---------------------------------------------------------------------------
# MultiTypeOr / MultiTypeAnd
# ---------------------------------------------------------------------------
class TestMultiTypeOrAnd:
    def test_or_repr(self) -> None:
        mt = MultiTypeOr(Numbers(), Strings())
        r = repr(mt)
        assert r.startswith("<MultiTypeOr:")
        assert r.endswith(">")

    def test_and_repr(self) -> None:
        mt = MultiTypeAnd(Numbers(min_value=0), Numbers(max_value=100))
        r = repr(mt)
        assert r.startswith("<MultiTypeAnd:")
        assert r.endswith(">")

    def test_and_valid_values_empty(self) -> None:
        mt = MultiTypeAnd(Numbers(), Ints())
        assert mt.valid_values == ()

    def test_or_validates_first_match(self) -> None:
        mt = MultiTypeOr(Numbers(), Strings())
        mt.validate(42)
        mt.validate("hello")

    def test_and_validates_all(self) -> None:
        mt = MultiTypeAnd(Numbers(min_value=0), Numbers(max_value=10))
        mt.validate(5)
        with pytest.raises(ValueError):
            mt.validate(-1)


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------
class TestArraysExtended:
    def test_unsupported_valid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="not supported"):
            Arrays(valid_types=[str])  # type: ignore[list-item]

    def test_complex_with_min_raises(self) -> None:
        with pytest.raises(TypeError, match="complex"):
            Arrays(valid_types=[np.complexfloating], min_value=0)

    def test_complex_with_max_raises(self) -> None:
        with pytest.raises(TypeError, match="complex"):
            Arrays(valid_types=[np.complexfloating], max_value=10)

    def test_min_greater_than_max_raises(self) -> None:
        with pytest.raises(TypeError, match="max_value must be bigger"):
            Arrays(min_value=10, max_value=1)

    def test_callable_shape(self) -> None:
        a = Arrays(shape=[lambda: 3, lambda: 2])
        arr = np.ones((3, 2))
        a.validate(arr)

    def test_shape_mismatch_raises(self) -> None:
        a = Arrays(shape=[2, 3])
        with pytest.raises(ValueError, match="shape"):
            a.validate(np.ones((4, 5)))

    def test_min_max_value_properties(self) -> None:
        a = Arrays(min_value=0, max_value=10)
        assert a.min_value == 0.0
        assert a.max_value == 10.0

    def test_min_max_value_none(self) -> None:
        a = Arrays()
        assert a.min_value is None
        assert a.max_value is None

    def test_validate_non_array_raises(self) -> None:
        a = Arrays()
        with pytest.raises(TypeError, match="not a numpy array"):
            a.validate([1, 2, 3])  # type: ignore[arg-type]

    def test_validate_wrong_dtype_raises(self) -> None:
        a = Arrays(valid_types=[np.integer])
        with pytest.raises(TypeError, match="is not any of"):
            a.validate(np.array([1.0, 2.0]))

    def test_max_value_exceeded(self) -> None:
        a = Arrays(min_value=0, max_value=5)
        with pytest.raises(ValueError, match="all values must be between"):
            a.validate(np.array([1, 2, 10]))

    def test_min_value_violated(self) -> None:
        a = Arrays(min_value=0, max_value=10)
        with pytest.raises(ValueError, match="all values must be between"):
            a.validate(np.array([-5, 2, 3]))

    def test_repr(self) -> None:
        a = Arrays(min_value=0, max_value=10, shape=[2, 3])
        r = repr(a)
        assert "Arrays" in r
        assert "shape" in r

    def test_shape_unevaluated_with_callable(self) -> None:
        fn = lambda: 5  # noqa: E731
        a = Arrays(shape=[fn, 3])
        raw = a.shape_unevaluated
        assert raw is not None
        assert raw[0] is fn
        assert raw[1] == 3

    def test_is_numeric(self) -> None:
        assert Arrays().is_numeric is True


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------
class TestListsExtended:
    def test_non_list_raises(self) -> None:
        lst = Lists()
        with pytest.raises(TypeError, match="not a list"):
            lst.validate((1, 2))  # type: ignore[arg-type]

    def test_with_element_validator(self) -> None:
        lst = Lists(elt_validator=Ints())
        lst.validate([1, 2, 3])
        with pytest.raises(TypeError):
            lst.validate(["a", "b"])

    def test_repr_format(self) -> None:
        lst = Lists(elt_validator=Ints())
        r = repr(lst)
        assert "Lists" in r
        assert "Ints" in r

    def test_elt_validator_property(self) -> None:
        iv = Ints()
        lst = Lists(elt_validator=iv)
        assert lst.elt_validator is iv

    def test_default_elt_validator_is_anything(self) -> None:
        lst = Lists()
        assert isinstance(lst.elt_validator, Anything)

    def test_empty_list_valid(self) -> None:
        lst = Lists(elt_validator=Ints())
        lst.validate([])


# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------
class TestSequenceExtended:
    def test_wrong_length_raises(self) -> None:
        s = Sequence(length=3)
        with pytest.raises(ValueError, match="length"):
            s.validate([1, 2])

    def test_correct_length_passes(self) -> None:
        s = Sequence(length=2)
        s.validate([1, 2])

    def test_unsorted_when_require_sorted_raises(self) -> None:
        s = Sequence(require_sorted=True)
        with pytest.raises(ValueError, match="sorted"):
            s.validate([3, 1, 2])

    def test_sorted_passes(self) -> None:
        s = Sequence(require_sorted=True)
        s.validate([1, 2, 3])

    def test_repr(self) -> None:
        s = Sequence(length=5, require_sorted=True)
        r = repr(s)
        assert "Sequence" in r
        assert "len: 5" in r
        assert "sorted: True" in r

    def test_properties(self) -> None:
        iv = Ints()
        s = Sequence(elt_validator=iv, length=4, require_sorted=True)
        assert s.elt_validator is iv
        assert s.length == 4
        assert s.require_sorted is True

    def test_non_sequence_raises(self) -> None:
        s = Sequence()
        with pytest.raises(TypeError, match="not a sequence"):
            s.validate(42)  # type: ignore[arg-type]

    def test_tuple_accepted(self) -> None:
        s = Sequence()
        s.validate((1, 2, 3))

    def test_with_element_validator(self) -> None:
        s = Sequence(elt_validator=Ints())
        s.validate([1, 2, 3])
        with pytest.raises(TypeError):
            s.validate(["a", "b"])


# ---------------------------------------------------------------------------
# Callable
# ---------------------------------------------------------------------------
class TestCallableExtended:
    def test_callable_passes(self) -> None:
        c = CallableValidator()
        c.validate(lambda: None)
        c.validate(len)

    def test_non_callable_raises(self) -> None:
        c = CallableValidator()
        with pytest.raises(TypeError, match="not a callable"):
            c.validate(42)

    def test_repr(self) -> None:
        assert repr(CallableValidator()) == "<Callable>"

    def test_valid_values_is_callable(self) -> None:
        c = CallableValidator()
        assert callable(c.valid_values[0])


# ---------------------------------------------------------------------------
# Dict
# ---------------------------------------------------------------------------
class TestDictExtended:
    def test_non_dict_raises(self) -> None:
        d = Dict()
        with pytest.raises(TypeError, match="not a dictionary"):
            d.validate([1, 2])  # type: ignore[arg-type]

    def test_forbidden_key_raises_syntax_error(self) -> None:
        d = Dict(allowed_keys=["a", "b"])
        with pytest.raises(SyntaxError, match="not in allowed keys"):
            d.validate({"a": 1, "c": 2})

    def test_allowed_keys_property(self) -> None:
        d = Dict(allowed_keys=["x", "y"])
        assert d.allowed_keys == ["x", "y"]

    def test_allowed_keys_setter(self) -> None:
        d = Dict()
        assert d.allowed_keys is None
        d.allowed_keys = ["a"]
        assert d.allowed_keys == ["a"]

    def test_repr_with_keys(self) -> None:
        d = Dict(allowed_keys=["a", "b"])
        r = repr(d)
        assert "Dict" in r
        assert "a" in r

    def test_repr_without_keys(self) -> None:
        assert repr(Dict()) == "<Dict>"

    def test_valid_dict_with_allowed_keys(self) -> None:
        d = Dict(allowed_keys=["a", "b"])
        d.validate({"a": 1, "b": 2})

    def test_any_dict_without_keys(self) -> None:
        d = Dict()
        d.validate({"anything": "goes", 42: True})

    def test_valid_values(self) -> None:
        d = Dict()
        assert d.valid_values == ({0: 1},)
