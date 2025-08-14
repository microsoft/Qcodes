from typing import ClassVar, Final

import pytest

from qcodes.extensions.parameters import ParameterMixin
from qcodes.parameters import Parameter, ParameterBase

#############################
# Common Classes for Testing
#############################


class CompatibleParameter(Parameter):
    """Docstring for CompatibleParameter"""

    pass


class AnotherCompatibleParameter(Parameter):
    """Docstring for AnotherCompatibleParameter"""

    pass


class IncompatibleParameter(Parameter):
    """Docstring for IncompatibleParameter"""

    pass


class CompatibleParameterMixin(ParameterMixin):
    """Docstring for CompatibleParameterMixin"""

    _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [CompatibleParameter]


class IntermediateParameterMixin(CompatibleParameterMixin):
    _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [
        CompatibleParameter,
        AnotherCompatibleParameter,
    ]


class LeafParameterMixin(IntermediateParameterMixin):
    _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [CompatibleParameter]


class AnotherCompatibleParameterMixin(ParameterMixin):
    """Docstring for AnotherCompatibleParameterMixin"""

    _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [CompatibleParameter]


class IncompatibleParameterMixin(ParameterMixin):
    """Docstring for IncompatibleParameterMixin"""

    _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [CompatibleParameter]
    _INCOMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [IncompatibleParameter]


#################################################
# Tests for Naming Conventions and Compatibility
#################################################


def test_correct_parameter_mixin_naming() -> None:
    class CorrectNamedParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [CompatibleParameter]

    assert CorrectNamedParameterMixin.__name__ == "CorrectNamedParameterMixin"


def test_incorrect_parameter_mixin_naming() -> None:
    with pytest.raises(ValueError):

        class IncorrectNamedMixin(ParameterMixin):
            _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [
                CompatibleParameter
            ]


def test_typeerror_if_multiple_parameterbase_leaves():
    class FakeParam1(ParameterBase):
        pass

    class FakeParam2(ParameterBase):
        pass

    with pytest.raises(TypeError):

        class BadParam(ParameterMixin, FakeParam1, FakeParam2):
            pass


def test_compatible_bases_must_be_list() -> None:
    with pytest.raises(TypeError):

        class IncorrectParameterMixin(ParameterMixin):
            _COMPATIBLE_BASES = "notalist"  # type: ignore[assignment]


def test_incompatible_bases_must_be_list() -> None:
    with pytest.raises(TypeError):

        class IncorrectParameterMixin(ParameterMixin):
            _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [
                CompatibleParameter
            ]
            _INCOMPATIBLE_BASES = IncompatibleParameter  # type: ignore[assignment]


##############################
# Tests for Docstring Updates
##############################


def test_docstring_update() -> None:
    class BaseParameter(Parameter):
        """Base Parameter Documentation."""

    class A_ParameterMixin(ParameterMixin):
        """A_Parameter Mixin Documentation."""

        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [BaseParameter]

    class B_ParameterMixin(ParameterMixin):
        """B_Parameter Mixin Documentation."""

        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [BaseParameter]

    class CombinedParameterMixin(A_ParameterMixin, B_ParameterMixin):
        _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True

    class CombinedParameter(CombinedParameterMixin, BaseParameter):
        """Combined Parameter Documentation."""

    doc = CombinedParameter.__doc__
    assert doc is not None
    assert "This Parameter has been extended by the following ParameterMixins" in doc
    assert "BaseParameter" in doc
    assert "Base Parameter Documentation." in doc
    assert "A_ParameterMixin" in doc
    assert "A_Parameter Mixin Documentation." in doc
    assert "B_ParameterMixin" in doc
    assert "B_Parameter Mixin Documentation." in doc
    assert "Combined Parameter Documentation." in doc


#################################
# Tests for Compatibility Checks
#################################


def test_warning_when_parameter_not_explicitly_compatible() -> None:
    with pytest.warns(UserWarning, match="is not explicitly compatible"):

        class TestParameter(CompatibleParameterMixin, AnotherCompatibleParameter):
            pass


def test_explicitly_compatible_base_parameter() -> None:
    class TestParameter(IntermediateParameterMixin, AnotherCompatibleParameter):
        pass


def test_leaf_parameter_mixin_with_another_base() -> None:
    with pytest.warns(UserWarning, match="is not explicitly compatible"):

        class TestParameter(LeafParameterMixin, AnotherCompatibleParameter):
            pass


def test_multiple_parameter_mixins_without_compatibility() -> None:
    class ComplexParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [
            AnotherCompatibleParameter
        ]

    with pytest.raises(
        TypeError, match="does not have any common compatible ParameterBase classes"
    ):

        class LeafComplexAParameterMixin(ComplexParameterMixin, LeafParameterMixin):
            _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True

    with pytest.warns(
        UserWarning, match="Multiple ParameterMixin are combined without"
    ):

        class LeafComplexBParameterMixin(ComplexParameterMixin, LeafParameterMixin):
            _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [
                CompatibleParameter
            ]


def test_sets_common_compatible_bases():
    class BaseA(ParameterBase):
        pass

    class BaseB(ParameterBase):
        pass

    class MixinAParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [BaseA, BaseB]

    class MixinBParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [BaseB]

    class ComboParameterMixin(MixinAParameterMixin, MixinBParameterMixin):
        _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True

    assert ComboParameterMixin._COMPATIBLE_BASES == [BaseB]


def test_multiple_mixins_no_compatible():
    class DummyBase(ParameterBase):
        pass

    class EmptyAParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = []

    class EmptyBParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = []

    # No compatible bases for either mixin
    with pytest.raises(TypeError):

        class ComboParameterMixin(EmptyAParameterMixin, EmptyBParameterMixin):
            _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = []
            _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True


def test_raises_if_incompatible():
    class BadBase(ParameterBase):
        pass

    class AParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = []
        _INCOMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [BadBase]

    with pytest.raises(TypeError, match="is incompatible with"):

        class Combo(AParameterMixin, BadBase):
            pass


def test_multiple_parameter_mixins_with_compatibility() -> None:
    class ComplexParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [
            AnotherCompatibleParameter
        ]

    class LeafComplexCParameterMixin(ComplexParameterMixin, LeafParameterMixin):
        _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True
        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [CompatibleParameter]

    class LeafComplexParameter(LeafComplexCParameterMixin, CompatibleParameter):
        pass


def test_multiple_parameter_mixins_explicit_compatibility_warnings() -> None:
    class ComplexParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [
            AnotherCompatibleParameter
        ]

    with pytest.warns(UserWarning, match="is not explicitly compatible"):

        class ComplexParameterA(ComplexParameterMixin, CompatibleParameter):
            pass


def test_mixing_complex_and_intermediate() -> None:
    class ComplexParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [
            AnotherCompatibleParameter
        ]

    # Warn when multiple ParameterMixins combined without declared compatibility
    with pytest.warns(
        UserWarning, match="Multiple ParameterMixin are combined without"
    ):

        class IntermediateComplexAParameterMixin(
            ComplexParameterMixin, IntermediateParameterMixin
        ):
            pass

    # Adding compatibility resolves this
    class IntermediateComplexBParameterMixin(
        ComplexParameterMixin, IntermediateParameterMixin
    ):
        _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True
        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [CompatibleParameter]

    class ComplexParameterB(IntermediateComplexBParameterMixin, CompatibleParameter):
        pass

    # Without explicit compatibility, warns again
    class IntermediateComplexCParameterMixin(
        ComplexParameterMixin, IntermediateParameterMixin
    ):
        _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True

    with pytest.warns(UserWarning, match="is not explicitly compatible"):

        class ComplexParameterC(
            IntermediateComplexCParameterMixin, CompatibleParameter
        ):
            pass

    # Using AnotherCompatibleParameter should now pass without warning
    class ComplexParameterD(
        IntermediateComplexCParameterMixin, AnotherCompatibleParameter
    ):
        pass


def test_typeerror_multiple_parameter_mixin_with_parameterbase():
    class MyBase(ParameterBase):
        pass

    class MyMixinAParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [MyBase]

    class MyMixinBParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [MyBase]

    with pytest.raises(TypeError):

        class Combo(MyMixinAParameterMixin, MyMixinBParameterMixin, MyBase):
            pass


#####################################################
# Tests for _get_leaf_classes and _get_mixin_classes
#####################################################


def test_get_leaf_classes_single_leaf() -> None:
    class TestParameter(LeafParameterMixin, CompatibleParameter):
        pass

    leaf_classes = TestParameter._get_leaf_classes(ParameterMixin, ParameterBase)
    assert LeafParameterMixin in leaf_classes
    assert IntermediateParameterMixin not in leaf_classes


def test_get_leaf_classes_multiple_leaves() -> None:
    class ComplexParameterMixin(AnotherCompatibleParameterMixin, LeafParameterMixin):
        _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True

    leaf_classes = ComplexParameterMixin._get_leaf_classes(
        ParameterMixin, ParameterBase
    )
    assert AnotherCompatibleParameterMixin in leaf_classes
    assert LeafParameterMixin in leaf_classes
    assert IntermediateParameterMixin not in leaf_classes
    assert ComplexParameterMixin not in leaf_classes


def test_get_leaf_classes_on_combined_parameter() -> None:
    class ComplexParameterMixin(AnotherCompatibleParameterMixin, LeafParameterMixin):
        _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True

    class ComplexParameterE(ComplexParameterMixin, CompatibleParameter):
        pass

    leaf_classes = ComplexParameterE._get_leaf_classes(ParameterMixin, ParameterBase)

    assert ComplexParameterMixin in leaf_classes

    assert CompatibleParameter not in leaf_classes
    assert AnotherCompatibleParameterMixin not in leaf_classes
    assert LeafParameterMixin not in leaf_classes
    assert IntermediateParameterMixin not in leaf_classes

    assert ParameterBase not in leaf_classes
    assert ParameterMixin not in leaf_classes


def test_get_mixin_classes() -> None:
    class ExtraParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [CompatibleParameter]

    class ComplexParameterMixin(ExtraParameterMixin, LeafParameterMixin):
        _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True

    class ComplexParameterF(ComplexParameterMixin, CompatibleParameter):
        pass

    all_mixins = ComplexParameterF._get_mixin_classes(ParameterMixin)

    assert ExtraParameterMixin in all_mixins
    assert CompatibleParameterMixin in all_mixins
    assert LeafParameterMixin in all_mixins
    assert IntermediateParameterMixin in all_mixins

    assert ComplexParameterF not in all_mixins

    assert ParameterBase not in all_mixins
    assert ParameterMixin not in all_mixins


def test_leaf_class_detection():
    class MyBase(ParameterBase):
        pass

    class MyMixinAParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = []

    class MyMixinBParameterMixin(MyMixinAParameterMixin):
        _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [MyBase]

    class MyParam(MyMixinBParameterMixin, MyBase):
        pass

    leafs = MyParam._get_leaf_classes(ParameterMixin, ParameterBase)
    assert MyMixinBParameterMixin in leafs
    assert MyMixinAParameterMixin not in leafs
