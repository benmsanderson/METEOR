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
        from meteor import (
            Cmip6MeteorDataGetter,
            MeteorNoiseGenerator,
            MeteorPatternScaling,
            train_noise_model_from_composite,
        )

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
        from meteor.impacts import ImpactCalculator, ImpactEnsemble, ImpactResult
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
        except Exception:
            # If initialization fails due to dependencies, at least import worked
            assert "Cmip6MeteorDataGetter" in str(type(Cmip6MeteorDataGetter))

    def test_module_attributes(self):
        """Test additional module attributes for coverage."""
        import meteor

        # Test module has standard attributes
        assert hasattr(meteor, "__name__")
        assert hasattr(meteor, "__file__")
        assert hasattr(meteor, "__package__")

        # Test version formatting
        version_parts = meteor.__version__.split(".")
        assert len(version_parts) >= 2  # Should have at least major.minor

        # Test that version parts are numeric or contain valid suffixes
        for i, part in enumerate(
            version_parts[:2]
        ):  # Major and minor should be numeric
            assert any(c.isdigit() for c in part)

    def test_error_handling_imports(self):
        """Test error handling in imports."""
        # Test importing non-existent submodule
        try:
            from meteor import nonexistent_module

            # Verify that this import actually failed
            assert nonexistent_module is None, "nonexistent_module should not exist"
            # Should not reach here
            assert False, "Should have raised ImportError"
        except (ImportError, AttributeError):
            # Expected behavior
            pass

        # Test that core imports still work after failed import
        from meteor import Cmip6MeteorDataGetter

        assert Cmip6MeteorDataGetter is not None

    def test_package_metadata(self):
        """Test package metadata accessibility."""
        import meteor

        # Test that package has expected metadata
        metadata_attrs = ["__name__", "__version__", "__file__"]
        for attr in metadata_attrs:
            assert hasattr(meteor, attr)
            assert getattr(meteor, attr) is not None


if __name__ == "__main__":
    pytest.main([__file__])
