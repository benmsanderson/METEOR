from token import ISNONTERMINAL
from eofs.xarray import Eof
import numpy as np
import lmfit
import re 
import xarray as xr
from scipy import signal


def make_anom(ds_4x,ds_cnt):
  #makes anomoly timeseries from 4xco2 and PICTRL
    ds_4x_anom=ds_4x-ds_cnt.mean("year")
    ds_4x_anom=ds_4x_anom.rename({'year': 'time'})
    ds_4x_anom=ds_4x_anom.interpolate_na(dim='lat', method='nearest').interpolate_na(dim='lon', method='nearest')
    return ds_4x_anom


def expotas(x, s1, t1):
  #single exponential decay function
  return s1*(1-np.exp(-x/t1))

def imodel_filter(pars, F, F0=7.41, y0=1850):
  #takes forcing timeseries as input and convolves with synthetic PC
  #timeseries generated from pars in response to 4xCO2
  #outputs a PC timeseries (xarray) corresponding to Forcing
  #timeseries and 4xCO2 EOFs
  #assumes default 4xCO2 forcing and output from 1850
  nt=len(F)
  dF=np.append(np.diff(F),0)/F0
  vals = pars.valuesdict()
  nm=len([value for key, value in vals.items() if 't' in key.lower()])
   
  us=pmodel(pars,nt)
  usa=xr.DataArray(us, coords=(np.arange(y0,nt+y0),np.arange(1,nm+1)), dims=('time','mode'))
  #Xrs=rmodel(eofout,usa)
  inm=np.apply_along_axis(lambda m: np.convolve(m, dF, mode='full'), axis=0, arr=usa)
  inm=inm[:nt]
  inma=xr.DataArray(inm, coords=(np.arange(y0,nt+y0),np.arange(0,nm)), dims=('time','mode'))

  return inma

def imodel_filter_scl(pscl, pars, F, F0=7.41, y0=1850):
  #applies quadratic scaling to forcing input to parameterise nonlinearities
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
  #makes synthetic PC timeseries from exponential curve parameters
    #nt=len(x)
    x=np.arange(0,nt)
    vals = pars.valuesdict()
    nm=len([value for key, value in vals.items() if 't' in key.lower()])
    aout=np.zeros([nt, nm])
    for i in np.arange(0,nm):
        for j in np.arange(0,nm):

            aout[:,i]=aout[:,i]+expotas(x,vals['s'+str(i)+str(j)],vals['t'+str(j)])
    return aout

def residual(pars, modewgt, data=None):
    #This returns the residual of synthetic minus true PC timeseries in response to a step change forcing
    wgtt=np.tile(modewgt.T,(data.shape[0],1))
    return wgtt*(data-pmodel(pars,data.shape[0]))

def residual_project_scl(pscl, pars, f, modewgt, data=None):
    #Takes a forcing timeseries (f), transforms with a quadratic using parameters in pscl and convolves with the empirical PCs generated from the exponential parameters in pars, returns residual compared with 'true' projected PCs in data
    wgtt=np.tile(modewgt.T,(data.shape[0],1))

    mdl=imodel_filter_scl(pscl, pars,f )
    rs=(data-mdl)
    rs=rs*wgtt
    return rs

def wgt(X):
    #returns latitudinal weights from X
    weights = np.cos(np.deg2rad(X.lat))
    weights.name = "weights"    
    return weights

def wgt2(X):
    #returns a lat-lon grid of weights and assigns slightly non-zero values to poles to prevent div by zero
    weights = wgt(X)
    wgt2=np.tile(weights,(len(X.lon),1)).T*.99+.01
    return wgt2

def makeparams_epc(t0=[1,30,1000]):
    #generates an lmfit parameter instance,
    #builds parameters governed by the number of intial estimates
    #given nm initial estimates, build empirical model for PCs of first nm modes
    #each mode is reconstructed as a sum of nm exponential decays
    #returns lmfit parameter instance
    #and adds timescale and amplitude parameters 
    nm=len(t0)
    fit_params = lmfit.Parameters()
    for i in np.arange(0,nm):
        fit_params.add('t'+str(i), value=t0[i],min=t0[i]/4,max=t0[i]*4)
        for j in np.arange(0,nm):

            fit_params.add('s'+str(i)+str(j), value=1)
    return fit_params


def svds(X,nm):
    #wrapper for the EOF package -returns modes and weights at truncation level nm
    solver = Eof(X,center=False,weights=wgt2(X))
    eofout= {}

    eofout['v']=solver.eofs(neofs=nm,eofscaling=1)
    eofout['u']=solver.pcs(npcs=nm,pcscaling=1)
    eofout['s']=solver.eigenvalues(neigs=nm)
    eofout['weights']=solver.getWeights()

    return eofout


def get_timescales(X,t0):
    #optimization for lmfit parameters
    nm=len(t0)
    nt=X.shape[0]
    eofout=svds(X,nm)
    x_array=np.arange(1,nt+1)

    fit_params=makeparams_epc(t0)
    modewgt=np.sqrt(eofout['s'])
    out = lmfit.minimize(residual, fit_params, args=(modewgt,), kws={'data': eofout['u']})
    #ts=[out.params['t1'].value,out.params['t2'].value,out.params['t3'].value]
    ts=[]
    for i in np.arange(0,nm):
        ts.append(out.params['t'+str(i)].value)
    us=pmodel(out.params,nt)
    usa=xr.DataArray(us, coords=(eofout['u'].time,eofout['u'].mode), dims=('time','mode'))
    return (ts,out,usa,eofout)

def makeparams_scl():
    #generates an lmfit parameter instance for quadratic scaling,

    scl_params = lmfit.Parameters()
    scl_params.add('a',value=.7,min=0.5,max=1.3)
    scl_params.add('b',value=0.043,min=0.0,max=0.5)

    return scl_params

def adjust_timescales(X,Xact,pars,t0,f):
    #optimization for quadratic forcing scaling
    scl_params = makeparams_scl(t0)
    
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
    #Reconstructs a time,lat,lon field from EOFs,PCs and weights
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



