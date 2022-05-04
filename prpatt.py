from eofs.xarray import Eof
import numpy as np
import lmfit
import re 

def make_4xanom(ds_4x,ds_cnt):
    ds_4x_anom=ds_4x-ds_cnt.mean("year")
    ds_4x_anom=ds_4x_anom.rename({'year': 'time'})
    ds_4x_anom=ds_4x_anom.interpolate_na(dim='lat', method='nearest').interpolate_na(dim='lon', method='nearest')
    return ds_4x_anom

def expotas(x, s1, s2, s3, t1, t2, t3):
    return s1*(1-np.exp(-x/t1))+s2*(1-np.exp(-x/t2))+s3*(1-np.exp(-x/t3))

def model(pars, x):
    vals = pars.valuesdict()

    a1=expotas(x, vals['s01'], vals['s02'], vals['s03'], vals['t1'], vals['t2'], vals['t3'])
    a2=expotas(x, vals['s11'], vals['s12'], vals['s13'], vals['t1'], vals['t2'], vals['t3'])
    a3=expotas(x, vals['s21'], vals['s22'], vals['s23'], vals['t1'], vals['t2'], vals['t3'])

    return np.stack((a1,a2,a3)).T

def residual(pars, x, data=None):
    return data-model(pars,x)
  
def wgt(X):
    weights = np.cos(np.deg2rad(X.lat))
    weights.name = "weights"
    wgt2=np.tile(weights,(len(X.lon),1)).T*.99+.01
    
    return wgt2

def get_timescales(X):
    nt=X.shape[0]
    solver = Eof(X,center=False,weights=wgt(X))

    v=solver.eofsAsCovariance(neofs=3)
    u=solver.pcs(npcs=3,pcscaling=1)
    s=solver.eigenvalues(neigs=3)

    x_array=np.arange(1,nt+1)

    fit_params = lmfit.Parameters()
    fit_params.add('t1', value=1)
    fit_params.add('t2', value=50)
    fit_params.add('t3', value=1000)
    fit_params.add('s01', value=1)
    fit_params.add('s02', value=1)
    fit_params.add('s03', value=1)
    fit_params.add('s11', value=1)
    fit_params.add('s12', value=1)
    fit_params.add('s13', value=1)
    fit_params.add('s21', value=1)
    fit_params.add('s22', value=1)
    fit_params.add('s23', value=1)

    out = lmfit.minimize(residual, fit_params, args=(x_array,), kws={'data': solver.pcs(npcs=3,pcscaling=1)})

def get_patterns(X,out):
    e1=expotas(x_array,1,0,0,out.params['t1'].value,0,0)
    e2=expotas(x_array,1,0,0,out.params['t2'].value,0,0)
    e3=expotas(x_array,1,0,0,out.params['t3'].value,0,0)

    us=np.stack((e1,e2,e3)).T
    u1=us/np.mean(us,0)

    v1f=np.dot(np.linalg.pinv(u1),X.values.reshape(nt,-1))
    v1=np.reshape(v1f,v.shape)

    #Xr=np.reshape(np.dot(u1,v1f),X.shape)         
    va=xr.DataArray(v1, coords=(None,X.lat,X.lon), dims=('mode','lat','lon'))
    
    return va


