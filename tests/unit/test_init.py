"""Tests for the main METEOR package initialization."""


def test_version_attribute():
    """Test that version attribute exists."""
    import meteor

    assert hasattr(meteor, "__version__")
    assert isinstance(meteor.__version__, str)
    assert len(meteor.__version__) > 0


def test_submodules_available():
    """Test that submodules are accessible."""
    import meteor

    # Test submodules exist
    assert hasattr(meteor, "cmip6_meteor_data_getter")
    assert hasattr(meteor, "meteor")
    assert hasattr(meteor, "noise_generator")
    assert hasattr(meteor, "pattern_logic_lib")
