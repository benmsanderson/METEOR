import os
import shutil

from meteor import cache_handling


def test_find_suitable_cache_location_dev(monkeypatch, tmp_path):
    # Simulate a development environment by creating a fake repo structure
    repo_root = tmp_path / "fake_repo"
    repo_root.mkdir()
    (repo_root / "setup.py").touch()  # Marker file

    # Create a subdirectory to simulate the location of the cache handler
    sub_dir = repo_root / "src" / "meteor"
    sub_dir.mkdir(parents=True)

    # Patch __file__ to point to a file in the subdirectory
    fake_file = sub_dir / "cache_handling.py"
    fake_file.touch()
    monkeypatch.setattr(cache_handling, "__file__", str(fake_file))

    cache_location = cache_handling.find_suitable_cache_location()
    expected_cache_location = repo_root / ".cache"
    assert cache_location == str(expected_cache_location)


def test_find_suitable_cache_location_pip(monkeypatch, tmp_path):
    # Simulate a pip-installed environment by creating a directory without marker files
    install_dir = tmp_path / "site-packages" / "meteor"
    install_dir.mkdir(parents=True)

    # Patch __file__ to point to a file in the install directory
    fake_file = install_dir / "cache_handling.py"
    fake_file.touch()
    monkeypatch.setattr(cache_handling, "__file__", str(fake_file))

    cache_location = cache_handling.find_suitable_cache_location()
    expected_cache_location = os.path.join(os.path.expanduser("~"), ".meteor", "cache")
    assert cache_location == expected_cache_location


def test_cache_handler_setup(tmp_path):
    cache_dir = tmp_path / "meteor_cache"
    handler = cache_handling.CacheHandler(cache_dir=str(cache_dir), purpose="classic")
    handler.setup_cache_tree()

    # Check that the main cache directory exists
    assert cache_dir.exists()
    # Check that the expected sub-caches exist
    expected_sub_caches = ["cmip6", "pattern_scaling"]
    for sub_cache in expected_sub_caches:
        sub_cache_path = cache_dir / sub_cache
        assert sub_cache_path.exists()

    assert handler.get_cmip6_query_catalogue() == os.path.join(
        cache_dir, "cmip6", "cmip6-zarr-consolidated-stores.csv"
    )
    shutil.rmtree(cache_dir)  # Clean up for next test
    handler_noise = cache_handling.CacheHandler(
        cache_dir=str(cache_dir), purpose="noise"
    )
    handler_noise.setup_cache_tree()
    expected_sub_caches = ["cmip6", "noise_models"]
    for sub_cache in expected_sub_caches:
        sub_cache_path = cache_dir / sub_cache
        assert sub_cache_path.exists()
    shutil.rmtree(cache_dir)  # Clean up for next test
    handler_general = cache_handling.CacheHandler(
        cache_dir=str(cache_dir), purpose="general"
    )
    handler_general.setup_cache_tree()
    expected_sub_caches = ["cmip6", "pattern_scaling", "noise_models"]
    for sub_cache in expected_sub_caches:
        sub_cache_path = cache_dir / sub_cache
        assert sub_cache_path.exists()

    shutil.rmtree(cache_dir)  # Clean up for next test
    handler_not_working = cache_handling.CacheHandler(cache_dir=expected_sub_caches)
    assert not handler_not_working.cache_functioning
    assert not handler_not_working.check_if_cmip6_cached(
        "get_single_var_mod_data_monthly", "piControl", "tas", "TestModel"
    )


def test_generate_cache_key():
    key1 = cache_handling._generate_cmip6_cache_key(
        "get_single_var_mod_data", "piControl", "tas", "CanESM5"
    )
    assert key1 == "CanESM5_piControl_tas_raw"
    key2 = cache_handling._generate_cmip6_cache_key(
        "get_single_var_mod_data_monthly", "piControl", "tas", "CanESM5"
    )
    assert key2 == "CanESM5_piControl_tas_monthly"
    key3 = cache_handling._generate_cmip6_cache_key(
        "get_single_var_mod_data_yearmean", "piControl", "tas", "CanESM5"
    )
    assert key3 == "CanESM5_piControl_tas_yearly"
    key4 = cache_handling._generate_cmip6_cache_key(
        "make_meteor_training_data", "piControl", "CanESM5", monthly=True
    )
    assert key4 == "CanESM5_piControl_training_monthly"
    key5 = cache_handling._generate_cmip6_cache_key(
        "make_meteor_training_data", "piControl", "CanESM5"
    )
    assert key5 == "CanESM5_piControl_training_yearly"
    key6 = cache_handling._generate_cmip6_cache_key(
        "make_meteor_training_data_composite",
        ["piControl", "historical"],
        "CanESM5",
        monthly=True,
    )
    assert key6 == "CanESM5_piControl_historical_composite_monthly"
    key7 = cache_handling._generate_cmip6_cache_key(
        "make_meteor_training_data_composite",
        ["piControl", "historical"],
        "CanESM5",
    )
    assert key7 == "CanESM5_piControl_historical_composite_yearly"
    key8 = cache_handling._generate_cmip6_cache_key(
        "unknown_method", "piControl", "tas", "CanESM5"
    )
    assert isinstance(key8, str)
    assert len(key8) == 32  # SHA-256 hash length


def test_find_expected_variable_from_args():
    var1 = cache_handling._find_expected_variable_from_args(
        "get_single_var_mod_data", "piControl", "tas", "CanESM5"
    )
    assert var1 == "tas"
    var2 = cache_handling._find_expected_variable_from_args(
        "get_single_var_mod_data_yearmean", "piControl", "pr", "CanESM5"
    )
    assert var2 == "pr"
    var3 = cache_handling._find_expected_variable_from_args(
        "get_single_var_mod_data_monthly", "piControl", "psl", "CanESM5"
    )
    assert var3 == "psl"
    var4 = cache_handling._find_expected_variable_from_args(
        "unknown_method", "piControl", "tas", "CanESM5"
    )
    assert var4 is None
    var5 = cache_handling._find_expected_variable_from_args(
        "get_single_var_mod_data", "piControl"
    )
    assert var5 is None


def test_cache_clearing(tmp_path):
    cache_dir = tmp_path / "meteor_cache"
    handler = cache_handling.CacheHandler(cache_dir=str(cache_dir), purpose="classic")
    handler.setup_cache_tree()

    # Create a dummy file in the cache
    dummy_file = cache_dir / "cmip6" / "dummy.nc"
    dummy_file.touch()
    assert dummy_file.exists()

    # Create a dummy file in the cache
    dummy_file2 = cache_dir / "pattern_scaling" / "dummy.nc"
    dummy_file2.touch()
    assert dummy_file.exists()

    # Clear the cache
    handler.clear_cache(sub_cache="cmip6")
    assert not dummy_file.exists()
    assert dummy_file2.exists()
    # Should do nothing if sub_cache does not exist
    handler.clear_cache(sub_cache="not_a_cache")
    assert dummy_file2.exists()
    handler.clear_cache()
    assert not dummy_file2.exists()


def test_cache_path_includes_variable_name():
    """Test that cache paths include variable name for tas and pr."""

    handler = cache_handling.CacheHandler(
        cache_dir="/tmp/meteor_cache", purpose="classic"
    )

    # Get cache paths for different variables
    tas_path = handler.get_pattern_scaling_cache_path("TestModel", variable="tas")
    pr_path = handler.get_pattern_scaling_cache_path("TestModel", variable="pr")
    print(tas_path)
    print(pr_path)
    # Verify different variables get different paths
    assert tas_path != pr_path, "tas and pr should have different cache paths"
    assert "tas" in tas_path, "tas cache path should contain 'tas'"
    assert "pr" in pr_path, "pr cache path should contain 'pr'"
    assert tas_path.endswith("_pattern_scaling.pkl")
    assert pr_path.endswith("_pattern_scaling.pkl")


def test_cache_path_without_variable():
    """Test that cache path works without variable for backward compatibility."""

    handler = cache_handling.CacheHandler(cache_dir="/tmp/", purpose="classic")
    assert handler.cache_functioning
    # Get cache path without variable parameter
    path_no_var = handler.get_pattern_scaling_cache_path("TestModel")
    print(path_no_var)
    # Should still work (for backward compatibility)
    assert "cmip6" in path_no_var
    assert path_no_var.endswith("_pattern_scaling.pkl")
    assert "TestModel" in path_no_var
