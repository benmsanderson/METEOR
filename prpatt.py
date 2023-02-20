from token import ISNONTERMINAL
from eofs.xarray import Eof
import numpy as np
import lmfit
import re 
import xarray as xr
from scipy import signal


def make_anom(ds_4x,ds_cnt):
  #makes anomoly timeseries from 4xco2 and PICTRL
    ds_4x_anom=ds_4x-ds_cnt.mean("year", skipna=True)
    ds_4x_anom=ds_4x_anom.rename({'year': 'time'})
    ds_4x_anom=ds_4x_anom.interpolate_na(dim='lat', method='nearest').interpolate_na(dim='lon', method='nearest')
    return ds_4x_anom


def expotas(x, s1, t1):
  #single exponential decay function
  return s1*(1-np.exp(-x/t1))

def imodel(pars, eofout, F, F0=7.41, y0=1850):
  #depreciated, reconstructs full grids and sums - very slow
  nt=len(F)
  
  us=pmodel(pars,nt)
  usa=xr.DataArray(us, coords=(np.arange(y0,nt+y0),eofout['u'].mode), dims=('time','mode'))
  Xrs=rmodel(eofout,usa)
  inm=Xrs*0.
  for Ft,i in enumerate(np.arange(0,nt-1)):
    dF=(F[i+1]-F[i])/F0
    ts=nt-i
    inm[i:,:,:]=inm[i:,:,:]+dF*Xrs[0:ts,:,:].values
  return inm

def imodel_eof(pars, F, F0=7.41, y0=1850):
  #depreciated - loop over forcing vecrtor, quite slow
  nt=len(F)
  vals = pars.valuesdict()
  nm=len([value for key, value in vals.items() if 'c' in key.lower()])
  us=pmodel(pars,nt)
  usa=xr.DataArray(us, coords=(np.arange(y0,nt+y0),np.arange(0,nm)), dims=('time','mode'))
  #Xrs=rmodel(eofout,usa)
  inm=usa*0.
  for Ft,i in enumerate(np.arange(0,nt-1)):
    dF=(F[i+1]-F[i])/F0
    ts=nt-i
    inm[i:,:]=inm[i:,:]+dF*usa[0:ts,:].values
  return inm

def imodel_filter(pars, F, F0=7.41, y0=1850):
  #takes forcing timeseries as input and convolves with synthetic PC
  #timeseries generated from pars in response to 4xCO2
  #outputs a PC timeseries (xarray) corresponding to Forcing
  #timeseries and 4xCO2 EOFs
  #assumes default 4xCO2 forcing and output from 1850
  nt=len(F)
  dF=np.append(np.diff(F),0)/F0
  vals = pars.valuesdict()
  nm=len([value for key, value in vals.items() if 'c' in key.lower()])
   
  us=pmodel(pars,nt)
  #print(us.shape)
  usa=xr.DataArray(us, coords=(np.arange(y0,nt+y0),np.arange(1,nm+1)), dims=('time','mode'))
  #Xrs=rmodel(eofout,usa)
  inm=np.apply_along_axis(lambda m: np.convolve(m, dF, mode='full'), axis=0, arr=usa)
  inm=inm[:nt]
  inma=xr.DataArray(inm, coords=(np.arange(y0,nt+y0),np.arange(0,nm)), dims=('time','mode'))

  return inma

def imodel_filter_scl(pscl, pars, F, F0=7.41, y0=1850):
  vals = pscl.valuesdict()
  fscl=vals['a']*F+vals['b']*np.square(F)
  inma=imodel_filter(pars, fscl, F0=7.41, y0=1850)
  return inma

def rmodel(eofout, us):
    #makes gridded field from EOFs and PC timeseries 'us'
  eof_synth=eofout.copy()
  eof_synth['u']=us
  Xrs=recon(eof_synth)
  return Xrs


def pmodel(pars, nt):
  #makes PC timeseries from parameters
    #nt=len(x)
    x=np.arange(0,nt)
    vals = pars.valuesdict()
    nm=len([value for key, value in vals.items() if 'c' in key.lower()])
    ntau=len([value for key, value in vals.items() if 't' in key.lower()])
    #print(nm)
    aout=np.zeros([nt, nm])
    for i in np.arange(0,nm):
        aout[:,i]=aout[:,i]+vals['c'+str(i)]
        for j in np.arange(0,ntau):
            aout[:,i]=aout[:,i]+expotas(x,vals['s'+str(j)+str(i)],vals['t'+str(j)])
    return aout

def residual(pars, modewgt, data=None):
    wgtt=np.tile(modewgt.T,(data.shape[0],1))
    return wgtt*(data-pmodel(pars,data.shape[0]))

def residual_project(pars, f, modewgt, data=None):
    wgtt=np.tile(modewgt.T,(data.shape[0],1))

    mdl=imodel_filter(pars,f )
    rs=(data-mdl)
    rs=rs*wgtt
    return rs

def residual_project_scl(pscl, pars, f, modewgt, data=None):
    wgtt=np.tile(modewgt.T,(data.shape[0],1))

    mdl=imodel_filter_scl(pscl, pars,f )
    rs=(data-mdl)
    rs=rs*wgtt
    return rs

def wgt(X):
    weights = np.cos(np.deg2rad(X.lat))
    weights.name = "weights"    
    return weights

def wgt2(X):
    weights = wgt(X)
    wgt2=np.tile(weights,(len(X.lon),1)).T*.99+.01
    return wgt2

def wgt3(X):
    weights = wgt(X)
    wgt3=np.tile(weights.T,(len(X.lon),len(X.year),1)).transpose([1,2,0])*.99+.01
    return wgt3

def gresid(pars,data=None):
    tmp=pmodel(pars,len(data)).squeeze()
    return tmp-data

def global_timescales(t0,data=None):
    pars=makeparams(t0,1)
    tms=global_mean(data).squeeze()
    out=lmfit.minimize(gresid, pars, kws={'data': tms})
    ts=[]
    for i in np.arange(0,len(t0)):
        ts.append(out.params['t'+str(i)].value)
    return ts

def makeparams(t0,nm):
    #nm=len(t0)
    nt=len(t0)
    fit_params = lmfit.Parameters()
    for i in np.arange(0,nt):
        fit_params.add('t'+str(i), value=t0[i],min=t0[i]/10,max=t0[i]*10)
        for j in np.arange(0,nm):

            fit_params.add('s'+str(i)+str(j), value=1)
    for j in np.arange(0,nm):
        fit_params.add('c'+str(j), value=0)
    return fit_params

def svds(X,nm):
    solver = Eof(X,center=False,weights=wgt2(X))
    eofout= {}

    eofout['v']=solver.eofs(neofs=nm,eofscaling=1)
    eofout['u']=solver.pcs(npcs=nm,pcscaling=1)
    eofout['s']=solver.eigenvalues(neigs=nm)
    eofout['weights']=solver.getWeights()

    return eofout
# calculate global means

def get_time_name(ds):
    for time_name in ['time', 'year']:
        if time_name in ds.coords:
            return time_name
    raise RuntimeError("Couldn't find a latitude coordinate")

def get_lat_name(ds):
    for lat_name in ['lat', 'latitude']:
        if lat_name in ds.coords:
            return lat_name

    raise RuntimeError("Couldn't find a latitude coordinate")

def global_mean(ds):
    lat = ds[get_lat_name(ds)]
    tm = get_time_name(ds)
    weight = np.cos(np.deg2rad(lat))
    weight = weight/weight.mean()
    other_dims = set(ds.dims) - {tm,'ens'}
    return (ds * weight).mean(other_dims)

def get_timescales(X,t0,nm):
    #nm=len(t0)
    nt=X.shape[0]
    eofout=svds(X,nm)
    x_array=np.arange(1,nt+1)

    fit_params=makeparams(t0,nm)
    modewgt=np.sqrt(eofout['s'])
    out = lmfit.minimize(residual, fit_params, args=(modewgt,), kws={'data': eofout['u']})
    #ts=[out.params['t1'].value,out.params['t2'].value,out.params['t3'].value]
    ts=[]
    for i in np.arange(0,len(t0)):
        ts.append(out.params['t'+str(i)].value)
    us=pmodel(out.params,nt)
    usa=xr.DataArray(us, coords=(eofout['u'].time,eofout['u'].mode), dims=('time','mode'))
    eofnew=eofout.copy()
    eofnew['u']=usa
    return (ts,out,usa,eofout,eofnew)

def adjust_timescales(X,Xact,pars,t0,f):
    scl_params = lmfit.Parameters()
    scl_params.add('a',value=1,min=0.7,max=1.3)
    scl_params.add('b',value=0.001,min=0.0,max=0.5)

    
    vals = pars.valuesdict()
    nm=len([value for key, value in vals.items() if 't' in key.lower()])
    nt=X.shape[0]
    solver = Eof(X,center=False,weights=wgt2(X))
    eofout=svds(X,nm)
    modewgt=np.sqrt(eofout['s'])

    ua=solver.projectField(Xact,neofs=nm,eofscaling=1)
    
    kys=[key for key, value in vals.items()]
    x_array=np.arange(1,nt+1)

    

    out = lmfit.minimize(residual_project_scl, scl_params, args=(pars,f,modewgt,), kws={'data': ua})
    #ts=[out.params['t1'].value,out.params['t2'].value,out.params['t3'].value]
    return out

def recon(eofout):
  u1=eofout['u']
  s1=eofout['s']
  v1=eofout['v']
  wgt=eofout['weights']

  nm=v1.shape[0]
  nt=u1.shape[0]
  v1f=v1.values.reshape(nm,-1)
  Xr=np.dot(np.dot(u1,np.diag(s1)),v1f)
  Xrp=np.reshape(Xr,[u1.shape[0],v1.shape[1],v1.shape[2]])/wgt
  Xo=xr.DataArray(Xrp, coords=(u1.time,v1.lat,v1.lon), dims=('time','lat','lon'))

  return Xo

def get_patterns_pinv(X,tsp):
    nt=X.shape[0]
    x_array=np.arange(1,nt+1)
    
    u0=np.empty([nt, len(tsp)])
    for i,ts in enumerate(tsp):
        tmp=expotas(x_array,1,ts)
        u0[:,i]=tmp/np.mean(tmp)

    #u1[:,-1]=u0[:,-1]-np.dot(np.dot(u0[:,-1],u0[:,:-1]),np.linalg.pinv(u0[:,:-1]))
    v1f=np.dot(np.linalg.pinv(u0),X.values.reshape(nt,-1))
    v1=np.reshape(v1f,(len(tsp),X.shape[1],X.shape[2]))

    #Xr=np.reshape(np.dot(u1,v1f),X.shape)         
    va=xr.DataArray(v1, coords=(np.array(tsp),X.lat,X.lon), dims=('mode','lat','lon'))
    
    return va,u1


