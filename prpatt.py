from eofs.xarray import Eof
import numpy as np
import lmfit
import re 
import xarray as xr
from scipy import signal


def make_anom(ds_4x,ds_cnt):
    ds_4x_anom=ds_4x-ds_cnt.mean("year")
    ds_4x_anom=ds_4x_anom.rename({'year': 'time'})
    ds_4x_anom=ds_4x_anom.interpolate_na(dim='lat', method='nearest').interpolate_na(dim='lon', method='nearest')
    return ds_4x_anom


def expotas(x, s1, t1):
  return s1*(1-np.exp(-x/t1))

def imodel(pars, eofout, F, F0=7.41, y0=1850):
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
  nt=len(F)
  vals = pars.valuesdict()
  nm=len([value for key, value in vals.items() if 't' in key.lower()])
   
  us=pmodel(pars,nt)
  usa=xr.DataArray(us, coords=(np.arange(y0,nt+y0),np.arange(1,nm+1)), dims=('time','mode'))
  #Xrs=rmodel(eofout,usa)
  inm=usa*0.
  for Ft,i in enumerate(np.arange(0,nt-1)):
    dF=(F[i+1]-F[i])/F0
    ts=nt-i
    inm[i:,:]=inm[i:,:]+dF*usa[0:ts,:].values
  return inm

def imodel_filter(pars, F, F0=7.41, y0=1850):
  nt=len(F)
  dF=np.append(np.diff(F),0)/F0
  vals = pars.valuesdict()
  nm=len([value for key, value in vals.items() if 't' in key.lower()])
   
  us=pmodel(pars,nt)
  usa=xr.DataArray(us, coords=(np.arange(y0,nt+y0),np.arange(1,nm+1)), dims=('time','mode'))
  #Xrs=rmodel(eofout,usa)
  inm=np.apply_along_axis(lambda m: np.convolve(m, dF, mode='full'), axis=0, arr=usa)
  inm=inm[:nt]
  return inm


def rmodel(eofout, us):
  eof_synth=eofout.copy()
  eof_synth['u']=us
  Xrs=recon(eof_synth)
  return Xrs
  #makes gridded pulse-response from parameters


def pmodel(pars, nt):
  #makes PC timeseries from parameters
    #nt=len(x)
    x=np.arange(0,nt)
    vals = pars.valuesdict()
    nm=len([value for key, value in vals.items() if 't' in key.lower()])
    aout=np.zeros([nt, nm])
    for i in np.arange(0,nm):
        for j in np.arange(0,nm):

            aout[:,i]=aout[:,i]+expotas(x,vals['s'+str(i)+str(j)],vals['t'+str(j)])
    return aout

def residual(pars, data=None):
    return data-pmodel(pars,data.shape[0])

def residual_project(pars, f, data=None):
    return data-imodel_filter(pars,f )

def wgt(X):
    weights = np.cos(np.deg2rad(X.lat))
    weights.name = "weights"    
    return weights

def wgt2(X):
    weights = wgt(X)
    wgt2=np.tile(weights,(len(X.lon),1)).T*.99+.01
    return wgt2

def makeparams(t0):
    nm=len(t0)
    fit_params = lmfit.Parameters()
    for i in np.arange(0,nm):
        fit_params.add('t'+str(i), value=t0[i],min=t0[i]/4,max=t0[i]*4)
        for j in np.arange(0,nm):

            fit_params.add('s'+str(i)+str(j), value=1)
    return fit_params

def svds(X,nm):
    solver = Eof(X,center=False,weights=wgt2(X))
    eofout= {}

    eofout['v']=solver.eofs(neofs=nm,eofscaling=1)
    eofout['u']=solver.pcs(npcs=nm,pcscaling=1)
    eofout['s']=solver.eigenvalues(neigs=nm)
    eofout['weights']=solver.getWeights()

    return eofout


def get_timescales(X,t0):
    nm=len(t0)
    nt=X.shape[0]
    eofout=svds(X,nm)
    x_array=np.arange(1,nt+1)

    fit_params=makeparams(t0)

    out = lmfit.minimize(residual, fit_params, args=(), kws={'data': eofout['u']})
    #ts=[out.params['t1'].value,out.params['t2'].value,out.params['t3'].value]
    ts=[]
    for i in np.arange(0,nm):
        ts.append(out.params['t'+str(i)].value)
    us=pmodel(out.params,nt)
    usa=xr.DataArray(us, coords=(eofout['u'].time,eofout['u'].mode), dims=('time','mode'))
    return (ts,out,usa,eofout)

def adjust_timescales(X,Xact,pars,t0,f):
    vals = pars.valuesdict()
    nm=len([value for key, value in vals.items() if 't' in key.lower()])
    nt=X.shape[0]
    solver = Eof(X,center=False,weights=wgt2(X))
    eofout=svds(X,nm)
    ua=solver.projectField(Xact,neofs=4,eofscaling=1)

    x_array=np.arange(1,nt+1)

    fit_params=pars

    out = lmfit.minimize(residual_project, fit_params, args=(f,), kws={'data': ua})
    #ts=[out.params['t1'].value,out.params['t2'].value,out.params['t3'].value]
    ts=[]
    for i in np.arange(0,nm):
        ts.append(out.params['t'+str(i)].value)
    us=pmodel(out.params,nt)
    usa=xr.DataArray(us, coords=(eofout['u'].time,eofout['u'].mode), dims=('time','mode'))
    return (ts,out,usa,eofout)

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


