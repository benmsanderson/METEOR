def make_anom(ds_scen,ds_cnt):
  #makes anomoly timeseries from 4xco2 and PICTRL
    ds_anom=ds_scen-ds_cnt.mean("year")
    ds_anom=ds_anom.rename({'year': 'time'})
    ds_anom=ds_anom.interpolate_na(dim='lat', method='nearest').interpolate_na(dim='lon', method='nearest')
    return ds_anom
