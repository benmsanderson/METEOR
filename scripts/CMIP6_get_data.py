#!/usr/bin/env python
# coding: utf-8

#Tested in Anaconda 3.7.13
#requires xesmf
#conda install -c conda-forge xesmf dask netCDF4 matplotlib cartopy jupyterlab zarr gcsfs

#from matplotlib import pyplot as plt

import numpy as np
import numpy.matlib
import pandas as pd
import xarray as xr
import zarr
import gcsfs
import pickle
import cftime
from sys import getsizeof
import time
from netCDF4 import num2date
import xesmf as xe

readdata=1
authdrive=1

datadir='/content/drive/MyDrive/colab_4xco2'

flds=['tas']
expts=['1pctCO2','piControl']#,'abrupt-4xCO2','historical','ssp126','ssp585']
calstrt=[True,True,True,False,False,False]
dbe=['CMIP','CMIP','CMIP','CMIP','ScenarioMIP','ScenarioMIP']

#lon_out=np.arange(1,359,2)
#lat_out=np.arange(-89,89,2)
#lons_sub, lats_sub = np.meshgrid(lon_out,lat_out)

grid_out=xe.util.grid_global(2,2)

#get_ipython().system('pip install --upgrade xarray zarr gcsfs cftime  nc-time-axis progress')





df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv', low_memory=False)


# Variables and experiments in database
vars=df.variable_id.unique()
vars.sort()
expts_full=df.experiment_id.unique()
expts_full.sort()

flds_full=df.variable_id.unique()
flds_full.sort()



# Make dataframe for each experiment type and each field
df_all1=[]
for i, row in enumerate(expts):
  df_ta1=[]
  for j,fld in enumerate(flds):
    tmp = df.query("activity_id=='"+dbe[i]+"' & table_id == 'Amon' & variable_id == '"+fld+"' & experiment_id == '"+expts[i]+"'")
    df_ta1.append(tmp)
  df_all1.append(df_ta1)


# Isolate unique models which have completed first experiment, create temporary model list


mdls1=df_all1[0][0].source_id.unique()
mdls1.sort()



# Make some empty dataframes to store concise list

# In[17]:


cnames=df_all1[0][0].columns
df_all=[]
for i, exp in enumerate(expts):
  tmp=[]
  for j,fld  in enumerate(flds):
    tmp.append(pd.DataFrame(columns=cnames))
  df_all.append(tmp)


# Now get 1 ensemble member for each model, if it exists, for each experiment.  Only add to dataframe df_ta if we have a full set of experiments.  Can't yet handle models not on a native lat/lon grid
print('Processing database...')
mdls=[]
n=0
for j, mdl in enumerate(mdls1):
    tmpdf=[]
    nruns=[]
    for i, ext in enumerate(expts):
        #find first variable for expt/model
        for j, fld in enumerate(flds):
          tmp=df_all1[i][j].query("source_id=='"+mdl+"'")
          nruns.append(tmp.shape[0])
    #is there at least 1 run per experiment,with all fields?
    if min(nruns)>=1:
      #point to the entry for 1st run, first variable for each expt
      for i, ext in enumerate(expts):
        mmb=df_all1[i][0]['member_id'].values[0]
        for j, fld in enumerate(flds):
          tt = df_all1[i][j].query("source_id=='"+mdl+"' & table_id == 'Amon'")
          df_all[i][j].loc[n]=tt.values[0]
      #add model to final list
      mdls.append(mdl)
      n=n+1
    
    


# Save model list as pickle


pickle.dump(mdls, open( datadir+"/mdls.pkl", "wb" ) )


# load Google cloud storage
gcs = gcsfs.GCSFileSystem(token='anon')


# Loop through zstore links, use zarr to open
# 

nm=len(mdls)
nf=len(flds)
ne=len(expts)

print('Reading zstore...')
if readdata:

  dsall=[]
  for i,df_ta in enumerate(df_all):
    dsm=[]
    for j,df in enumerate(df_ta):
      ds=[]
      print(expts[i]+','+flds[j])
      for index, item in enumerate(df.zstore.values, start=0):
        mapper=gcs.get_mapper(item)
        ds.append(xr.open_zarr(mapper, decode_times=False))
      dsm.append(ds)
    dsall.append(dsm)  


# concatenated dataarrays for ts, global mean
print('Processing data...')
if readdata:
 dall=[]

 for i,ds in enumerate(dsall,start=0):
   dexp=[]
   for j,dm in enumerate(ds):
     print(expts[i]+','+flds[j])
     for index, dd in enumerate(dm, start=0):
         if 'longitude' in dd.keys():
           dd=dd.rename({'longitude': 'lon','latitude': 'lat'})
         if 'latitude' in dd.coords:
           dd=dd.drop('latitude')  
           dd=dd.drop_dims('latitude')
      
         #tmp=dd[flds[j]][:4800,:,:].interp(lon=lon_out,lat=lat_out, kwargs={"fill_value": "extrapolate"})
         
         if calstrt[i]:
           tmp.coords['time']=pd.date_range('1850-01-01', periods=tmp['time'].values.shape[0],freq='M')
         if tmp['time'].dtype=='float64' or  tmp['time'].dtype=='int64':
           tmp.coords['time']=num2date(tmp['time'].values,tmp['time'].units)     
         srm=tmp.groupby('time.year').mean('time')
         if index==0:
           dac=srm
         else:
           dac=xr.concat([dac,srm],'ens',coords='minimal',compat='override')
     dexp.append(dac)
   dall.append(dexp)

print('Postprocessing output...')
ci=expts.index('piControl')
tmp_ctrl=xr.merge(dall[ci][:]).mean("year")

if readdata:
  for i,d in enumerate(dall,start=0):
    print(expts[i])
    tmp=xr.merge(d[:])
    ds_anom=tmp-tmp_ctrl
    ds_anom=ds_anom.rename({'year': 'time'})
    ds_anom=ds_anom.interpolate_na(dim='lat', method='nearest').interpolate_na(dim='lon', method='nearest')
    ds_anom.to_netcdf(datadir+'/colab_4xco2/'+expts[i]+'_anom.nc')




