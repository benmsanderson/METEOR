from eofs.xarray import Eof
import numpy as np
import lmfit
import re 
import xarray as xr

def make_4xanom(ds_4x,ds_cnt):
    ds_4x_anom=ds_4x-ds_cnt.mean("year")
    ds_4x_anom=ds_4x_anom.rename({'year': 'time'})
    ds_4x_anom=ds_4x_anom.interpolate_na(dim='lat', method='nearest').interpolate_na(dim='lon', method='nearest')
    return ds_4x_anom


def expotas(x, s1, t1):
  return s1*(1-np.exp(-x/t1))

def model(pars, x):
    nt=len(x)
    vals = pars.valuesdict()
    nm=len([value for key, value in vals.items() if 't' in key.lower()])
    aout=np.zeros([nt, nm])
    for i in np.arange(0,nm):
        for j in np.arange(0,nm):

            aout[:,i]=aout[:,i]+expotas(x,vals['s'+str(i)+str(j)],vals['t'+str(j)])
    return aout

def residual(pars, x, nm, data=None):
    return data-model(pars,x)

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
        fit_params.add('t'+str(i), value=t0[i],min=0)
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

    out = lmfit.minimize(residual, fit_params, args=(x_array,nm,), kws={'data': eofout['u']})
    #ts=[out.params['t1'].value,out.params['t2'].value,out.params['t3'].value]
    ts=[]
    for i in np.arange(0,nm):
        ts.append(out.params['t'+str(i)].value)
    us=model(out.params,np.arange(0,nt))
    return (ts,out,us,eofout)

    eofout=svds(X,nm)

def recon(eofout):
  u1=eofout['u']
  s1=eofout['s']
  v1=eofout['v']
  wgt=eofout['weights']

  nm=v1.shape[0]
  nt=u1.shape[0]
  v1f=v1.values.reshape(nm,-1)
  Xr=np.dot(np.dot(u1,np.diag(np.sqrt(s1))),v1f)
  Xrp=np.reshape(Xr,[u1.shape[0],v1.shape[1],v1.shape[2]])
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


