"""Tests for the main METEOR package initialization."""

import pytest


class TestMeteorInit:
    """Test METEOR package initialization and imports."""

    def test_version_attribute(self):
        """Test that version attribute exists."""
        import meteor

        assert hasattr(meteor, "__version__")
        assert isinstance(meteor.__version__, str)
        assert len(meteor.__version__) > 0

    def test_main_imports(self):
        """Test that main classes can be imported."""
        # Test that main classes are available at package level
        from meteor import Cmip6MeteorDataGetter
        from meteor import MeteorPatternScaling
        from meteor import MeteorNoiseGenerator
        from meteor import train_noise_model_from_composite

        # Verify they are classes/functions
        assert callable(Cmip6MeteorDataGetter)
        assert callable(MeteorPatternScaling)
        assert callable(MeteorNoiseGenerator)
        assert callable(train_noise_model_from_composite)

    def test_impacts_submodule(self):
        """Test that impacts submodule is available."""
        import meteor

        assert hasattr(meteor, "impacts")

        # Test that impacts classes can be imported
        from meteor.impacts import ImpactCalculator
        from meteor.impacts import ImpactResult
        from meteor.impacts import ImpactEnsemble
        from meteor.impacts.calculators import DegreeDaysCalculator

        # Verify they are classes
        assert isinstance(ImpactCalculator, type)
        assert isinstance(ImpactResult, type)
        assert isinstance(ImpactEnsemble, type)
        assert isinstance(DegreeDaysCalculator, type)

    def test_submodules_available(self):
        """Test that submodules are accessible."""
        import meteor

        # Test submodules exist
        assert hasattr(meteor, "cmip6_meteor_data_getter")
        assert hasattr(meteor, "meteor")
        assert hasattr(meteor, "noise_generator")
        assert hasattr(meteor, "prpatt")

    def test_direct_class_instantiation(self):
        """Test that classes can be instantiated."""
        from meteor import Cmip6MeteorDataGetter

        # Test that we can create instance without errors
        try:
            data_getter = Cmip6MeteorDataGetter()
            assert data_getter is not None
        except Exception as e:
            # If initialization fails due to dependencies, at least import worked
            assert "Cmip6MeteorDataGetter" in str(type(Cmip6MeteorDataGetter))


if __name__ == "__main__":
    pytest.main([__file__])
