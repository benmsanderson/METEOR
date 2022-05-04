from eofs.xarray import Eof
import numpy as np
import lmfit
import re 

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
    nm=int(vals['nm'])
    aout=np.empty([nt, nm])
    for i in np.arange(0,nm):
        for j in np.arange(0,nm):
            aout[:,i]=aout[:,i]+expotas(x,vals['s'+str(i)+str(j)],vals['t'+str(i)])
    return aout

def residual(pars, x, data=None):
    return data-model(pars,x)
  
def wgt(X):
    weights = np.cos(np.deg2rad(X.lat))
    weights.name = "weights"
    wgt2=np.tile(weights,(len(X.lon),1)).T*.99+.01
    
    return wgt2

def makeparams(t0):
    nm=len(t0)
    fit_params = lmfit.Parameters()
    for i in np.arange(0,nm):
        fit_params.add('t'+str(i), value=t0[i])
        for j in np.arange(0,nm):
            print(i,j)
            fit_params.add('s'+str(i)+str(j), value=1)
    fit_params.add('nm',value=nm)
    return fit_params

def get_timescales(X,t0):
    nm=len(t0)
    nt=X.shape[0]
    solver = Eof(X,center=False,weights=wgt(X))

    v=solver.eofsAsCovariance(neofs=nm)
    u=solver.pcs(npcs=nm,pcscaling=1)
    s=solver.eigenvalues(neigs=nm)

    x_array=np.arange(1,nt+1)

    fit_params=makeparams(t0)

    out = lmfit.minimize(residual, fit_params, args=(x_array,), kws={'data': solver.pcs(npcs=nm,pcscaling=1)})
    #ts=[out.params['t1'].value,out.params['t2'].value,out.params['t3'].value]
    return out

def get_patterns(X,tsp):
    nt=X.shape[0]
    x_array=np.arange(1,nt+1)
    
    u1=np.empty([nt, len(tsp)])
    for i,ts in enumerate(tsp):
        tmp=expotas(x_array,1,ts)
        u1[:,i]=tmp/np.mean(tmp)

    v1f=np.dot(np.linalg.pinv(u1),X.values.reshape(nt,-1))
    v1=np.reshape(v1f,(len(tsp),X.shape[1],X.shape[2]))

    #Xr=np.reshape(np.dot(u1,v1f),X.shape)         
    va=xr.DataArray(v1, coords=(np.array(tsp),X.lat,X.lon), dims=('mode','lat','lon'))
    
    return va,u1


