"""
Variable-Specific Transform Registry

This module defines transformations that should be applied to specific
climate variables to ensure physical realism (e.g., precipitation positivity).
"""

from meteor.precipitation_transform import (
    fit_distribution_parameters_1d,
    fit_distribution_parameters_3d,
    apply_distribution_transform,
)


class VariableTransformConfig:
    """
    Configuration for variable-specific transformations.
    
    Parameters
    ----------
    name : str
        Name of the transform
    transform_type : str or None
        Type of distribution transform ('gamma', 'lognorm', etc.) or None
    reason : str
        Human-readable explanation of why this transform is needed
    fit_1d_func : callable, optional
        Function to fit 1D transform parameters
    fit_3d_func : callable, optional
        Function to fit 3D transform parameters
    apply_func : callable, optional
        Function to apply the transform
    """
    
    def __init__(self, name, transform_type, reason, 
                 fit_1d_func=None, fit_3d_func=None, apply_func=None):
        self.name = name
        self.transform_type = transform_type
        self.reason = reason
        self.fit_1d_func = fit_1d_func
        self.fit_3d_func = fit_3d_func
        self.apply_func = apply_func
    
    def __repr__(self):
        if self.transform_type is None:
            return "VariableTransformConfig(None - no transform)"
        return f"VariableTransformConfig('{self.transform_type}' - {self.reason})"


# Registry of variable-specific transforms
VARIABLE_TRANSFORMS = {
    'pr': VariableTransformConfig(
        name='precipitation_gamma',
        transform_type='gamma',
        reason='Ensure positive-only values (precipitation cannot be negative)',
        fit_1d_func=fit_distribution_parameters_1d,
        fit_3d_func=fit_distribution_parameters_3d,
        apply_func=apply_distribution_transform
    ),
    'tas': VariableTransformConfig(
        name=None,
        transform_type=None,
        reason='No transform needed for temperature',
    ),
    # Future variable transforms can be added here:
    # 'hurs': VariableTransformConfig(
    #     name='bounded_0_100',
    #     transform_type='beta',
    #     reason='Ensure 0-100% bounds for relative humidity',
    # ),
    # 'clt': VariableTransformConfig(
    #     name='bounded_0_100',
    #     transform_type='beta',
    #     reason='Ensure 0-100% bounds for cloud fraction',
    # ),
    # 'sfcWind': VariableTransformConfig(
    #     name='wind_speed_positive',
    #     transform_type='gamma',
    #     reason='Ensure positive-only values for wind speed',
    # ),
}


def get_variable_transform_config(variable):
    """
    Get the default transform configuration for a variable.
    
    Parameters
    ----------
    variable : str
        Variable name (e.g., 'tas', 'pr')
    
    Returns
    -------
    VariableTransformConfig
        Transform configuration for the variable, or identity transform if unknown
    """
    if variable in VARIABLE_TRANSFORMS:
        return VARIABLE_TRANSFORMS[variable]
    else:
        # Default: no transform for unknown variables
        return VariableTransformConfig(
            name=None,
            transform_type=None,
            reason=f'Unknown variable {variable} - no transform applied'
        )


def list_available_transforms():
    """Print all available variable transforms."""
    print("Available Variable Transforms:")
    print("=" * 60)
    for var, config in VARIABLE_TRANSFORMS.items():
        if config.transform_type:
            print(f"  {var:10s} : {config.transform_type:10s} - {config.reason}")
        else:
            print(f"  {var:10s} : (none)       - {config.reason}")
    print("=" * 60)
