import pytest
from typing import Final
from qcodes.parameters import ParameterBase, Parameter, ParameterMixin
from qcodes.validators import Numbers


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
    _COMPATIBLE_BASES = [CompatibleParameter]

class IntermediateParameterMixin(CompatibleParameterMixin):
    _COMPATIBLE_BASES = [CompatibleParameter, AnotherCompatibleParameter]

class LeafParameterMixin(IntermediateParameterMixin):
    _COMPATIBLE_BASES = [CompatibleParameter]

class AnotherCompatibleParameterMixin(ParameterMixin):
    """Docstring for AnotherCompatibleParameterMixin"""
    _COMPATIBLE_BASES = [CompatibleParameter]

class IncompatibleParameterMixin(ParameterMixin):
    """Docstring for IncompatibleParameterMixin"""
    _COMPATIBLE_BASES = [CompatibleParameter]
    _INCOMPATIBLE_BASES = [IncompatibleParameter]


#################################################
# Tests for Naming Conventions and Compatibility
#################################################

def test_correct_parameter_mixin_naming():
    class CorrectNamedParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES = [CompatibleParameter]

    assert CorrectNamedParameterMixin.__name__ == "CorrectNamedParameterMixin"


def test_incorrect_parameter_mixin_naming():
    with pytest.raises(ValueError):
        class IncorrectNamedMixin(ParameterMixin):
            _COMPATIBLE_BASES = [CompatibleParameter]


def test_compatible_bases_must_be_list():
    with pytest.raises(TypeError):
        class IncorrectParameterMixin(ParameterMixin):
            _COMPATIBLE_BASES = CompatibleParameter


def test_incompatible_bases_must_be_list():
    with pytest.raises(TypeError):
        class IncorrectParameterMixin(ParameterMixin):
            _COMPATIBLE_BASES = [CompatibleParameter]
            _INCOMPATIBLE_BASES = IncompatibleParameter


##############################
# Tests for Docstring Updates
##############################

def test_docstring_update():
    class BaseParameter(Parameter):
        """Base Parameter Documentation."""

    class A_ParameterMixin(ParameterMixin):
        """A_Parameter Mixin Documentation."""
        _COMPATIBLE_BASES = [BaseParameter]

    class B_ParameterMixin(ParameterMixin):
        """B_Parameter Mixin Documentation."""
        _COMPATIBLE_BASES = [BaseParameter]

    class CombinedParameterMixin(A_ParameterMixin, B_ParameterMixin):   
        _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True

    class CombinedParameter(CombinedParameterMixin, BaseParameter):
        """Combined Parameter Documentation."""

    doc = CombinedParameter.__doc__
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

def test_warning_when_parameter_not_explicitly_compatible():
    with pytest.warns(UserWarning, match="is not explicitly compatible"):
        class TestParameter(CompatibleParameterMixin, AnotherCompatibleParameter):
            pass


def test_explicitly_compatible_base_parameter():
    class TestParameter(IntermediateParameterMixin, AnotherCompatibleParameter):
        pass


def test_leaf_parameter_mixin_with_another_base():
    with pytest.warns(UserWarning, match="is not explicitly compatible"):
        class TestParameter(LeafParameterMixin, AnotherCompatibleParameter):
            pass


def test_multiple_parameter_mixins_without_compatibility():
    class ComplexParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES = [AnotherCompatibleParameter]

    with pytest.raises(TypeError, match="does not have any common compatible ParameterBase classes"):
        class LeafComplexParameterMixin(ComplexParameterMixin, LeafParameterMixin):
            _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True

    with pytest.warns(UserWarning, match="Multiple ParameterMixin are combined without"):
        class LeafComplexParameterMixin(ComplexParameterMixin, LeafParameterMixin):
            _COMPATIBLE_BASES = [CompatibleParameter]


def test_multiple_parameter_mixins_with_compatibility():
    class ComplexParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES = [AnotherCompatibleParameter]

    class LeafComplexParameterMixin(ComplexParameterMixin, LeafParameterMixin):
        _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True
        _COMPATIBLE_BASES = [CompatibleParameter]

    class LeafComplexParameter(LeafComplexParameterMixin, CompatibleParameter):
        pass


def test_multiple_parameter_mixins_explicit_compatibility_warnings():
    class ComplexParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES = [AnotherCompatibleParameter]

    with pytest.warns(UserWarning, match="is not explicitly compatible"):
        class ComplexParameter(ComplexParameterMixin, CompatibleParameter):
            pass


def test_mixing_complex_and_intermediate():
    class ComplexParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES = [AnotherCompatibleParameter]

    # Warn when multiple ParameterMixins combined without declared compatibility
    with pytest.warns(UserWarning, match="Multiple ParameterMixin are combined without"):
        class IntermediateComplexParameterMixin(ComplexParameterMixin, IntermediateParameterMixin):
            pass

    # Adding compatibility resolves this
    class IntermediateComplexParameterMixin(ComplexParameterMixin, IntermediateParameterMixin):
        _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True
        _COMPATIBLE_BASES = [CompatibleParameter]

    class ComplexParameter(IntermediateComplexParameterMixin, CompatibleParameter):
        pass

    # Without explicit compatibility, warns again
    class IntermediateComplexParameterMixin(ComplexParameterMixin, IntermediateParameterMixin):
        _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True

    with pytest.warns(UserWarning, match="is not explicitly compatible"):
        class ComplexParameter(IntermediateComplexParameterMixin, CompatibleParameter):
            pass

    # Using AnotherCompatibleParameter should now pass without warning
    class ComplexParameter(IntermediateComplexParameterMixin, AnotherCompatibleParameter):
        pass


#####################################################
# Tests for _get_leaf_classes and _get_mixin_classes
#####################################################

def test_get_leaf_classes_single_leaf():
    class TestParameter(LeafParameterMixin, CompatibleParameter):
        pass

    leaf_classes = TestParameter._get_leaf_classes(ParameterMixin, ParameterBase)
    assert LeafParameterMixin in leaf_classes
    assert IntermediateParameterMixin not in leaf_classes


def test_get_leaf_classes_multiple_leaves():
    class ComplexParameterMixin(AnotherCompatibleParameterMixin, LeafParameterMixin):
        _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True

    leaf_classes = ComplexParameterMixin._get_leaf_classes(ParameterMixin, ParameterBase)
    assert AnotherCompatibleParameterMixin in leaf_classes
    assert LeafParameterMixin in leaf_classes
    assert IntermediateParameterMixin not in leaf_classes
    assert ComplexParameterMixin not in leaf_classes


def test_get_leaf_classes_on_combined_parameter():
    class ComplexParameterMixin(AnotherCompatibleParameterMixin, LeafParameterMixin):
        _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True

    class ComplexParameter(ComplexParameterMixin, CompatibleParameter):
        pass

    leaf_classes = ComplexParameter._get_leaf_classes(ParameterMixin, ParameterBase)
    
    assert ComplexParameterMixin in leaf_classes

    assert CompatibleParameter not in leaf_classes
    assert AnotherCompatibleParameterMixin not in leaf_classes
    assert LeafParameterMixin not in leaf_classes
    assert IntermediateParameterMixin not in leaf_classes
    
    assert ParameterBase not in leaf_classes
    assert ParameterMixin not in leaf_classes


def test_get_mixin_classes():
    class ExtraParameterMixin(ParameterMixin):
        _COMPATIBLE_BASES = [CompatibleParameter]

    class ComplexParameterMixin(ExtraParameterMixin, LeafParameterMixin):
        _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True

    class ComplexParameter(ComplexParameterMixin, CompatibleParameter):
        pass

    all_mixins = ComplexParameter._get_mixin_classes(ParameterMixin)

    assert ExtraParameterMixin in all_mixins
    assert CompatibleParameterMixin in all_mixins
    assert LeafParameterMixin in all_mixins
    assert IntermediateParameterMixin in all_mixins

    assert ComplexParameter not in all_mixins

    assert ParameterBase not in all_mixins
    assert ParameterMixin not in all_mixins
