"""
Tests for variable-specific transform configurations.

Tests the VariableTransformConfig class and the variable transform registry.
"""

import io
import sys

from meteor.variable_transforms import (
    VARIABLE_TRANSFORMS,
    VariableTransformConfig,
    get_variable_transform_config,
    list_available_transforms,
)


def test_initialization_minimal():
    """Test basic initialization with minimal parameters."""
    config = VariableTransformConfig(
        name="test_transform", transform_type="gamma", reason="Test reason"
    )

    assert config.name == "test_transform"
    assert config.transform_type == "gamma"
    assert config.reason == "Test reason"
    assert config.fit_1d_func is None
    assert config.fit_3d_func is None
    assert config.apply_func is None


def test_initialization_with_functions():
    """Test initialization with transform functions."""

    def dummy_fit_1d():
        pass

    def dummy_fit_3d():
        pass

    def dummy_apply():
        pass

    config = VariableTransformConfig(
        name="test_transform",
        transform_type="gamma",
        reason="Test reason",
        fit_1d_func=dummy_fit_1d,
        fit_3d_func=dummy_fit_3d,
        apply_func=dummy_apply,
    )

    assert config.fit_1d_func is dummy_fit_1d
    assert config.fit_3d_func is dummy_fit_3d
    assert config.apply_func is dummy_apply


def test_initialization_none_transform():
    """Test initialization with no transform (None)."""
    config = VariableTransformConfig(
        name=None, transform_type=None, reason="No transform needed"
    )

    assert config.name is None
    assert config.transform_type is None
    assert config.reason == "No transform needed"


def test_repr_with_transform():
    """Test __repr__ with a transform type."""
    config = VariableTransformConfig(
        name="gamma_transform",
        transform_type="gamma",
        reason="Ensure positive values",
    )

    repr_str = repr(config)

    assert "VariableTransformConfig" in repr_str
    assert "gamma" in repr_str
    assert "Ensure positive values" in repr_str


def test_repr_without_transform():
    """Test __repr__ with no transform."""
    config = VariableTransformConfig(
        name=None, transform_type=None, reason="No transform needed"
    )

    repr_str = repr(config)

    assert "VariableTransformConfig" in repr_str
    assert "None - no transform" in repr_str


# Test get_variable_transform_config() function.


def test_get_tas_config():
    """Test getting transform config for temperature (tas)."""
    config = get_variable_transform_config("tas")

    assert isinstance(config, VariableTransformConfig)
    assert config.name is None
    assert config.transform_type is None
    assert "No transform needed" in config.reason


def test_get_pr_config():
    """Test getting transform config for precipitation (pr)."""
    config = get_variable_transform_config("pr")

    assert isinstance(config, VariableTransformConfig)
    assert config.name == "precipitation_gamma"
    assert config.transform_type == "gamma"
    assert (
        "positive-only" in config.reason.lower()
        or "cannot be negative" in config.reason.lower()
    )
    # Should have callable functions
    assert callable(config.fit_1d_func)
    assert callable(config.fit_3d_func)
    assert callable(config.apply_func)


def test_get_unknown_variable_config():
    """Test getting transform config for unknown variable."""
    config = get_variable_transform_config("unknown_var")

    assert isinstance(config, VariableTransformConfig)
    assert config.name is None
    assert config.transform_type is None
    assert "unknown_var" in config.reason.lower()
    assert "no transform" in config.reason.lower()


def test_get_config_returns_from_registry():
    """Test that known variables return registry configs."""
    # Get config for pr
    config_pr = get_variable_transform_config("pr")

    # Should be the same object as in registry
    assert config_pr is VARIABLE_TRANSFORMS["pr"]


def test_get_config_creates_new_for_unknown():
    """Test that unknown variables get new config instances."""
    config1 = get_variable_transform_config("unknown1")
    config2 = get_variable_transform_config("unknown2")

    # Should be different instances
    assert config1 is not config2
    assert "unknown1" in config1.reason
    assert "unknown2" in config2.reason


# Test the VARIABLE_TRANSFORMS registry.


def test_registry_exists():
    """Test that registry is defined."""
    assert VARIABLE_TRANSFORMS is not None
    assert isinstance(VARIABLE_TRANSFORMS, dict)


def test_tas_in_registry():
    """Test that tas is in registry."""
    assert "tas" in VARIABLE_TRANSFORMS
    config = VARIABLE_TRANSFORMS["tas"]
    assert config.transform_type is None


def test_pr_in_registry():
    """Test that pr is in registry with gamma transform."""
    assert "pr" in VARIABLE_TRANSFORMS
    config = VARIABLE_TRANSFORMS["pr"]
    assert config.transform_type == "gamma"
    assert config.fit_1d_func is not None
    assert config.fit_3d_func is not None
    assert config.apply_func is not None


def test_registry_configs_are_instances():
    """Test that all registry entries are VariableTransformConfig instances."""
    for var, config in VARIABLE_TRANSFORMS.items():
        assert isinstance(
            config, VariableTransformConfig
        ), f"Config for {var} is not a VariableTransformConfig instance"


# Test list_available_transforms() function.


def test_list_available_transforms_runs():
    """Test that list_available_transforms() executes without error."""
    # Should not raise any exceptions
    # try:

    # Capture print output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    list_available_transforms()

    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__

    # Verify output contains expected content
    assert "Available Variable Transforms" in output
    assert "tas" in output
    assert "pr" in output
    assert "=" in output  # Header separator


# except Exception as e:
#     pytest.fail(f"list_available_transforms() raised an exception: {e}")
# finally:
#     sys.stdout = sys.__stdout__


def test_list_available_transforms_shows_pr_transform():
    """Test that pr's gamma transform is shown in output."""
    captured_output = io.StringIO()
    sys.stdout = captured_output

    list_available_transforms()

    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__

    # pr should show gamma transform
    assert "pr" in output
    assert "gamma" in output


def test_list_available_transforms_shows_tas_none():
    """Test that tas shows no transform in output."""
    captured_output = io.StringIO()
    sys.stdout = captured_output

    list_available_transforms()

    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__

    # tas should show (none)
    assert "tas" in output
    assert "(none)" in output or "No transform" in output
