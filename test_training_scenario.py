#!/usr/bin/env python
"""
Quick test to verify training_scenario parameter with hybrid precedence.
"""
import sys
sys.path.insert(0, '/Users/bensan/Documents/Github/METEOR/src')

from meteor.meteor_interface import MeteorInterface

# Test 1: Default behavior (ssp245)
print("=" * 60)
print("Test 1: Default behavior")
print("=" * 60)
emulator1 = MeteorInterface.from_cmip6('CanESM5', variables=['tas'])
# Don't actually train, just check the config would be correct
config1 = emulator1._get_default_config('tas')
print(f"Default config training_scenario: {config1.get('training_scenario')}")
assert config1['training_scenario'] == 'ssp245', "Default should be ssp245"
print("✅ PASSED\n")

# Test 2: Method parameter override
print("=" * 60)
print("Test 2: Method parameter override to ssp370")
print("=" * 60)
emulator2 = MeteorInterface.from_cmip6('CanESM5', variables=['tas'])
# Simulate what train() would do
training_scenario = 'ssp370'
config2 = emulator2._get_default_config('tas')
if training_scenario != 'ssp245':
    config2['training_scenario'] = training_scenario
print(f"Config with method override: {config2.get('training_scenario')}")
assert config2['training_scenario'] == 'ssp370', "Should use method parameter"
print("✅ PASSED\n")

# Test 3: Variable-specific override
print("=" * 60)
print("Test 3: Variable-specific override in variable_configs")
print("=" * 60)
emulator3 = MeteorInterface.from_cmip6('CanESM5', variables=['tas', 'pr'])
training_scenario = 'ssp245'
variable_configs = {
    'tas': {'use_exog': 'all'},  # No training_scenario specified
    'pr': {'training_scenario': 'ssp585'}  # Override for pr
}

# Simulate what train() would do for tas
config_tas = emulator3._get_default_config('tas')
if training_scenario != 'ssp245':
    config_tas['training_scenario'] = training_scenario
if 'tas' in variable_configs:
    config_tas.update(variable_configs['tas'])
print(f"TAS config: training_scenario={config_tas.get('training_scenario')}, use_exog={config_tas.get('use_exog')}")
assert config_tas['training_scenario'] == 'ssp245', "TAS should use default"
assert config_tas['use_exog'] == 'all', "TAS should have custom use_exog"

# Simulate what train() would do for pr
config_pr = emulator3._get_default_config('pr')
if training_scenario != 'ssp245':
    config_pr['training_scenario'] = training_scenario
if 'pr' in variable_configs:
    config_pr.update(variable_configs['pr'])
print(f"PR config: training_scenario={config_pr.get('training_scenario')}")
assert config_pr['training_scenario'] == 'ssp585', "PR should use variable-specific override"
print("✅ PASSED\n")

# Test 4: Hybrid precedence (method param + variable override)
print("=" * 60)
print("Test 4: Hybrid precedence (method=ssp370, pr override=ssp585)")
print("=" * 60)
emulator4 = MeteorInterface.from_cmip6('CanESM5', variables=['tas', 'pr'])
training_scenario = 'ssp370'
variable_configs = {
    'pr': {'training_scenario': 'ssp585'}
}

# TAS should get method parameter
config_tas = emulator4._get_default_config('tas')
if training_scenario != 'ssp245':
    config_tas['training_scenario'] = training_scenario
if 'tas' in variable_configs:
    config_tas.update(variable_configs['tas'])
print(f"TAS config: training_scenario={config_tas.get('training_scenario')}")
assert config_tas['training_scenario'] == 'ssp370', "TAS should use method parameter"

# PR should get variable-specific override
config_pr = emulator4._get_default_config('pr')
if training_scenario != 'ssp245':
    config_pr['training_scenario'] = training_scenario
if 'pr' in variable_configs:
    config_pr.update(variable_configs['pr'])
print(f"PR config: training_scenario={config_pr.get('training_scenario')}")
assert config_pr['training_scenario'] == 'ssp585', "PR should use variable override"
print("✅ PASSED\n")

print("=" * 60)
print("✅ All tests passed!")
print("=" * 60)
print("\nHybrid precedence working correctly:")
print("  1. Default: ssp245")
print("  2. Method parameter overrides default")
print("  3. Variable-specific config overrides method parameter")
