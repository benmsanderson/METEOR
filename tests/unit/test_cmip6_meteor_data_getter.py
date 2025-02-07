import pytest
import xarray as xr

from meteor import cmip6_meteor_data_getter, prpatt
import matplotlib.pyplot as plt


def test_get_unique_models():
    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter()
    models = data_getter.get_models_avail()
    print(models)
    assert isinstance(models, list)
    assert len(models) > 0

    assert data_getter.check_if_model_has_data("CanESM5")
    assert not data_getter.check_if_model_has_data("NorESM1")

    with pytest.raises(KeyError, match="No or incomplete data for NorESM1"):
        data_getter.get_single_var_mod_data_yearmean("piControl", "tas", "NorESM1")
    with pytest.raises(
        KeyError,
        match="This datagetter does not handle data from the historical experiment, available options are *",
    ):
        data_getter.get_single_var_mod_data_yearmean("historical", "tas", "CanESM5")
    with pytest.raises(
        KeyError,
        match="This datagetter does not handle rsut data, available options are *",
    ):
        data_getter.get_single_var_mod_data_yearmean("piControl", "rsut", "CanESM5")

    # data_getter.get_single_var_mod_data("CanESM5", "piControl", "tas")
    assert isinstance(data_getter.df_all, list)
    test_var = data_getter.get_single_var_mod_data_yearmean(
        "piControl", "tas", "CanESM5"
    )
    assert isinstance(test_var, xr.DataArray)
    test_training = data_getter.make_meteor_training_data("base", "CanESM5")
    print(test_training)
    assert isinstance(test_training, xr.Dataset)

    data_getter_2 = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        exps=["historical", "ssp370"], dbe=["CMIP", "ScenarioMIP"]
    )
    test_composite = data_getter_2.make_meteor_training_data_composite(
        ["historical", "ssp370"], model="CanESM5"
    )
    assert test_composite.sizes["year"] == 251
    print(test_composite["year"].values)
    # assert False


def test_make_meteor_training_data_composite_more_than_one():
    exps = ["historical", "ssp585", "ssp534-over"]
    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        exps=exps, dbe=["CMIP", "ScenarioMIP", "ScenarioMIP"]
    )
    models = data_getter.models
    print(len(models))

    #test_composite = data_getter.make_meteor_training_data_composite(
    #    exps, model="CanESM5", overlap={"ssp534-over": "Full-back"}
    #)
    #assert len(test_composite["year"].values) == 251

    #test_composite = data_getter.make_meteor_training_data_composite(
    #    exps, model="CanESM5", overlap={"ssp534-over": 61}
    #)
    #assert len(test_composite["year"].values) == 251
    weird_models = {}
    for model in models:
        print(model)

        ssp585 = data_getter.get_single_var_mod_data_yearmean("ssp585", "tas", model)
        ssp534 = data_getter.get_single_var_mod_data_yearmean("ssp534-over", "tas", model)
        
        print(len(ssp585["year"]))
        print(len(ssp534["year"]))
        years_585 = range(2015, 2015 + len(ssp585["year"]))
        years_534 = range(2040, 2040 + len(ssp534["year"]))
        mean_ssp585 = prpatt.global_mean(ssp585)
        mean_ssp534 = prpatt.global_mean(ssp534)
        print(mean_ssp585.values[0][ 23:27])
        print(mean_ssp534.values[0][:3])

        fig = plt.subplot()


        fig.plot(years_585, mean_ssp585.values[0], ls=":")
        fig.plot(years_534, mean_ssp534.values[0], ls=":")
        fig.set_title(f"Global mean temperature for {model}")
        fig.set_xlabel("Year")
        fig.set_ylabel("Global mean temperature")
        

        try: 
            test_composite = data_getter.make_meteor_training_data_composite(
                exps, model=model, overlap={"ssp534-over": "Full-back"}
            )
            composite_mean = prpatt.global_mean(test_composite["tas"])
            print(composite_mean)
            years_composite = range(1850, 1850 + len(composite_mean["year"]))
            fig.plot(years_composite, composite_mean.values[0], ls=":")
        except:
            print(f"Model {model} has a problem generating the composite")
            weird_models[model] = [len(ssp585["year"]), len(ssp534["year"])]

        plt.savefig(f"test_composite_temperature_{model}.png")
        plt.clf()

        print(weird_models)
    assert False