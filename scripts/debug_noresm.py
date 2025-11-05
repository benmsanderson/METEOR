#!/usr/bin/env python3
"""
Test script to debug NorESM2-MM SSP availability issue
"""

import pandas as pd

def test_noresm_availability():
    """Test what's available for NorESM2-MM in different activity_ids"""
    
    print("Loading CMIP6 catalog...")
    df = pd.read_csv(
        "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv",
        low_memory=False,
    )
    
    model = "NorESM2-MM"
    experiment = "ssp585"
    variable = "tas"
    
    print(f"\nTesting {model} - {experiment} - {variable}")
    print("="*50)
    
    # Test with CMIP activity_id (current default)
    cmip_query = df.query(
        f"activity_id=='CMIP' & table_id == 'Amon' & variable_id == '{variable}' & experiment_id == '{experiment}' & source_id == '{model}'"
    )
    print(f"With activity_id='CMIP': {len(cmip_query)} results")
    if len(cmip_query) > 0:
        print(f"  Found: {cmip_query[['source_id', 'experiment_id', 'variable_id', 'member_id']].values}")
    
    # Test with ScenarioMIP activity_id (correct for SSP)
    scenario_query = df.query(
        f"activity_id=='ScenarioMIP' & table_id == 'Amon' & variable_id == '{variable}' & experiment_id == '{experiment}' & source_id == '{model}'"
    )
    print(f"With activity_id='ScenarioMIP': {len(scenario_query)} results")
    if len(scenario_query) > 0:
        print(f"  Found: {scenario_query[['source_id', 'experiment_id', 'variable_id', 'member_id']].values}")
    
    # Show what activity_ids are available for this experiment
    exp_activities = df.query(f"experiment_id == '{experiment}'")['activity_id'].unique()
    print(f"\nAll activity_ids available for {experiment}: {list(exp_activities)}")
    
    # Show what experiments NorESM2-MM has in ScenarioMIP
    noresm_scenarios = df.query(f"source_id == '{model}' & activity_id == 'ScenarioMIP'")['experiment_id'].unique()
    print(f"\nNorESM2-MM experiments in ScenarioMIP: {list(noresm_scenarios)}")

if __name__ == "__main__":
    test_noresm_availability()