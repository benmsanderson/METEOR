#!/usr/bin/env python3
"""
METEOR-impacts Demo Script

This script demonstrates how to use the new METEOR-impacts layer to calculate
cooling and heating degree days from climate model output.

The METEOR-impacts system provides:
1. Extensible impact calculator framework
2. DegreeDaysCalculator implementing Isaac & van Vuuren (2009) methodology
3. Ensemble processing capabilities
4. Integration with existing METEOR/METEOR-noise workflow

Author: METEOR Team
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from meteor.impacts import DegreeDaysCalculator, ImpactEnsemble, ensemble_statistics


def create_synthetic_climate_data():
    """Create synthetic monthly temperature data similar to METEOR output."""
    print("Creating synthetic climate data...")
    
    # Simulate 20 years of monthly temperature data
    n_years = 20
    n_months = n_years * 12
    
    # Small spatial grid for demonstration
    lats = np.array([35, 45, 55])  # Three latitude bands
    lons = np.array([0, 10, 20, 30])  # Four longitude points
    
    # Create time axis
    months = np.arange(n_months)
    years = months // 12
    
    # Create realistic temperature patterns
    temperature_data = np.zeros((n_months, len(lats), len(lons)))
    
    for i, month in enumerate(months):
        year = month // 12
        month_of_year = month % 12
        
        # Base temperature varies with latitude (colder at higher latitudes)
        base_temp = 20 - 0.6 * (lats - 35)  # 20°C at 35N, 8°C at 55N
        
        # Seasonal cycle (amplitude varies with latitude)
        seasonal_amplitude = 12 + 0.3 * (lats - 35)  # Larger cycles at higher latitudes
        seasonal_cycle = seasonal_amplitude * np.cos(2 * np.pi * (month_of_year - 6) / 12)
        
        # Add some climate change warming trend (2°C over 20 years)
        warming_trend = 2.0 * year / n_years
        
        # Combine temperature components
        monthly_temp = (base_temp[:, np.newaxis] + 
                       seasonal_cycle[:, np.newaxis] + 
                       warming_trend)
        
        # Add random weather variability
        monthly_temp = monthly_temp + np.random.normal(0, 2, (len(lats), len(lons)))
        
        temperature_data[i, :, :] = monthly_temp
    
    # Create xarray DataArray
    temperature_da = xr.DataArray(
        temperature_data,
        dims=['month', 'lat', 'lon'],
        coords={
            'month': months,
            'lat': lats,
            'lon': lons,
        },
        attrs={
            'long_name': 'Monthly Mean Temperature',
            'units': 'degC',
            'description': 'Synthetic climate data for METEOR-impacts demo'
        }
    )
    
    print(f"Created temperature data: {temperature_da.shape}")
    print(f"Temperature range: {temperature_da.min().values:.1f}°C to {temperature_da.max().values:.1f}°C")
    
    return temperature_da


def demonstrate_single_calculation():
    """Demonstrate basic degree days calculation for a single realization."""
    print("\n" + "="*60)
    print("DEMONSTRATION 1: Single Climate Realization")
    print("="*60)
    
    # Create climate data
    temperature_data = create_synthetic_climate_data()
    
    # Initialize calculator
    calculator = DegreeDaysCalculator(base_temperature=18.0)
    print(f"Using base temperature: {calculator.base_temperature}°C")
    
    # Calculate degree days
    print("Calculating degree days...")
    result = calculator.calculate(temperature_data)
    
    # Display results
    print(f"Results contain: {list(result.keys())}")
    
    # Show annual totals
    annual_hdd = result['annual_hdd']
    annual_cdd = result['annual_cdd']
    
    print(f"\nAnnual HDD range: {annual_hdd.min().values:.0f} to {annual_hdd.max().values:.0f} degree-days")
    print(f"Annual CDD range: {annual_cdd.min().values:.0f} to {annual_cdd.max().values:.0f} degree-days")
    
    # Calculate spatial means
    hdd_spatial_mean = annual_hdd.mean(dim=['lat', 'lon'])
    cdd_spatial_mean = annual_cdd.mean(dim=['lat', 'lon'])
    
    print(f"\nTime series of spatial mean HDD: min={hdd_spatial_mean.min().values:.0f}, max={hdd_spatial_mean.max().values:.0f}")
    print(f"Time series of spatial mean CDD: min={cdd_spatial_mean.min().values:.0f}, max={cdd_spatial_mean.max().values:.0f}")
    
    # Show trend due to warming
    hdd_trend = np.polyfit(range(len(hdd_spatial_mean)), hdd_spatial_mean.values, 1)[0]
    cdd_trend = np.polyfit(range(len(cdd_spatial_mean)), cdd_spatial_mean.values, 1)[0]
    
    print(f"\nClimate trends per year:")
    print(f"  HDD: {hdd_trend:.1f} degree-days/year (decreasing with warming)")
    print(f"  CDD: {cdd_trend:.1f} degree-days/year (increasing with warming)")
    
    return result


def demonstrate_ensemble_processing():
    """Demonstrate ensemble processing with multiple climate realizations."""
    print("\n" + "="*60)
    print("DEMONSTRATION 2: Climate Ensemble Processing")
    print("="*60)
    
    # Create multiple climate realizations
    n_members = 5
    ensemble_data = []
    
    print(f"Creating {n_members} ensemble members...")
    for i in range(n_members):
        # Set different random seed for each member
        np.random.seed(42 + i)
        member_data = create_synthetic_climate_data()
        ensemble_data.append(member_data)
    
    # Reset random seed
    np.random.seed()
    
    # Process ensemble using ImpactEnsemble class
    calculator = DegreeDaysCalculator(base_temperature=18.0)
    impact_ensemble = ImpactEnsemble(calculator)
    
    print("Processing ensemble...")
    ensemble_results = impact_ensemble.calculate_ensemble(ensemble_data)
    
    print(f"Ensemble contains {len(ensemble_results)} members")
    
    # Calculate ensemble statistics
    print("Calculating ensemble statistics...")
    
    # Calculate statistics for HDD and CDD separately
    hdd_stats = ensemble_statistics(ensemble_results, 'annual_hdd')
    cdd_stats = ensemble_statistics(ensemble_results, 'annual_cdd')
    
    print(f"HDD statistics available: {list(hdd_stats.data_vars)}")
    print(f"CDD statistics available: {list(cdd_stats.data_vars)}")
    
    # Show ensemble spread
    hdd_mean = hdd_stats['ensemble_mean'].mean(dim=['lat', 'lon']).mean()
    hdd_std = hdd_stats['ensemble_std'].mean(dim=['lat', 'lon']).mean()
    cdd_mean = cdd_stats['ensemble_mean'].mean(dim=['lat', 'lon']).mean()
    cdd_std = cdd_stats['ensemble_std'].mean(dim=['lat', 'lon']).mean()
    
    print(f"\nEnsemble statistics (spatial and temporal averages):")
    print(f"  HDD: {hdd_mean.values:.0f} ± {hdd_std.values:.0f} degree-days")
    print(f"  CDD: {cdd_mean.values:.0f} ± {cdd_std.values:.0f} degree-days")
    
    # Show uncertainty range
    hdd_min = hdd_stats['ensemble_min'].mean(dim=['lat', 'lon']).mean()
    hdd_max = hdd_stats['ensemble_max'].mean(dim=['lat', 'lon']).mean()
    cdd_min = cdd_stats['ensemble_min'].mean(dim=['lat', 'lon']).mean()
    cdd_max = cdd_stats['ensemble_max'].mean(dim=['lat', 'lon']).mean()
    
    print(f"\nUncertainty ranges (min-max across ensemble):")
    print(f"  HDD: {hdd_min.values:.0f} to {hdd_max.values:.0f} degree-days")
    print(f"  CDD: {cdd_min.values:.0f} to {cdd_max.values:.0f} degree-days")
    
    return hdd_stats, cdd_stats


def demonstrate_base_temperature_sensitivity():
    """Demonstrate sensitivity to different base temperatures."""
    print("\n" + "="*60)
    print("DEMONSTRATION 3: Base Temperature Sensitivity")
    print("="*60)
    
    # Create climate data
    temperature_data = create_synthetic_climate_data()
    
    # Test different base temperatures
    base_temperatures = [15.0, 18.0, 21.0, 24.0]
    results = {}
    
    for base_temp in base_temperatures:
        print(f"Calculating for base temperature: {base_temp}°C")
        calculator = DegreeDaysCalculator(base_temperature=base_temp)
        result = calculator.calculate(temperature_data)
        results[base_temp] = result
    
    # Compare results
    print(f"\nBase temperature sensitivity (spatial and temporal averages):")
    print(f"{'Base Temp (°C)':<15} {'Mean HDD':<12} {'Mean CDD':<12} {'Total DD':<12}")
    print("-" * 55)
    
    for base_temp in base_temperatures:
        hdd_mean = results[base_temp]['annual_hdd'].mean().values
        cdd_mean = results[base_temp]['annual_cdd'].mean().values
        total_dd = hdd_mean + cdd_mean
        
        print(f"{base_temp:<15.1f} {hdd_mean:<12.0f} {cdd_mean:<12.0f} {total_dd:<12.0f}")
    
    print(f"\nObservations:")
    print(f"  - Lower base temperatures result in more CDD, less HDD")
    print(f"  - Higher base temperatures result in more HDD, less CDD")
    print(f"  - Total degree days varies with base temperature")
    print(f"  - Choice of base temperature affects absolute values but trends remain")


def main():
    """Run all demonstrations."""
    print("METEOR-IMPACTS DEMONSTRATION")
    print("="*60)
    print("This script demonstrates the new METEOR-impacts layer for calculating")
    print("climate impact metrics from temperature data.")
    print()
    print("The impacts layer implements the Isaac & van Vuuren (2009) methodology")
    print("for estimating cooling and heating degree days from monthly temperature.")
    print()
    print("Key features:")
    print("  - Extensible impact calculator framework")
    print("  - Robust degree days calculation with proper error handling")
    print("  - Ensemble processing and uncertainty quantification")
    print("  - Integration with existing METEOR workflow")
    
    # Run demonstrations
    try:
        result1 = demonstrate_single_calculation()
        result2_hdd, result2_cdd = demonstrate_ensemble_processing()
        demonstrate_base_temperature_sensitivity()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("The METEOR-impacts system is now ready for use!")
        print()
        print("To use in your own code:")
        print("  from meteor.impacts import DegreeDaysCalculator")
        print("  calculator = DegreeDaysCalculator(base_temperature=18.0)")
        print("  result = calculator.calculate(your_temperature_data)")
        print()
        print("For ensemble processing:")
        print("  from meteor.impacts import ImpactEnsemble, ensemble_statistics")
        print("  ensemble = ImpactEnsemble(calculator)")
        print("  results = ensemble.calculate_ensemble(ensemble_data)")
        print("  stats = ensemble_statistics(results)")
        
    except Exception as e:
        print(f"\nERROR during demonstration: {e}")
        print("Please check that METEOR-impacts is properly installed.")
        raise


if __name__ == "__main__":
    main()