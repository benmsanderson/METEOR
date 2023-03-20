"""
PRPATT
"""
import lmfit
import numpy as np
import xarray as xr
from eofs.xarray import Eof

# This is a library of functions to provide the backend to the pulse-response logic in METEOR


def make_anom(ds_4x, ds_cnt):
    # makes anomoly timeseries from 4xco2 and PICTRL
    ds_4x_anom = ds_4x - ds_cnt.mean("year", skipna=True)
    ds_4x_anom = ds_4x_anom.rename({"year": "time"})
    ds_4x_anom = ds_4x_anom.interpolate_na(dim="lat", method="nearest").interpolate_na(
        dim="lon", method="nearest"
    )
    return ds_4x_anom


def expotas(x, s1, t1):
    """
    Return single exponential pulse response function

    At time x value of exponential pulse response function with
    time scale t1 and coefficient s1

    Parameters
    ----------
    x : float
        time at which to evalute function
    s1 : float
         Coefficent of decay function
    t1 : float
         Decay time scale
    """
    return s1 * (1 - np.exp(-x / t1))


def imodel_filter(pars, F, F0=7.41, y0=1850):
    # takes forcing timeseries as input and convolves with synthetic PC
    # timeseries generated from input parameters
    # (derived in response to forcing step function)
    # The function outputs a PC timeseries (xarray) corresponding to Forcing
    # timeseries and EOFs patterns
    # F0 is the step function forcing size (defaults to 4xCO2 level)
    # y0 is the first year in the output xarray (defaults 1850)

    # length of forcing timeseries
    nt = len(F)
    # vector of forcing differences, dF - append 0
    dF = np.append(np.diff(F), 0) / F0
    # get parameter value dictionary
    vals = pars.valuesdict()
    # how many modes are present in the parameters? - define nm
    nm = len([value for key, value in vals.items() if "c" in key.lower()])
    # create the synthetic pulse-response kernel for a unit step function
    # output needs to be nt in length - long enough for the first timestep of the convolution
    us = pmodel(pars, nt)
    # print(us.shape)
    # make the PC timeseries into an xarray Dataarray
    usa = xr.DataArray(
        us, coords=(np.arange(y0, nt + y0), np.arange(1, nm + 1)), dims=("time", "mode")
    )
    # Convolution step - convolve the forcing difference timeseries dF with the step function kernel
    inm = np.apply_along_axis(
        lambda m: np.convolve(m, dF, mode="full"), axis=0, arr=usa
    )
    # truncate the output to the length of the forcing timeseries
    inm = inm[:nt]
    # format the output as a datarray
    inma = xr.DataArray(
        inm, coords=(np.arange(y0, nt + y0), np.arange(0, nm)), dims=("time", "mode")
    )
    # return convolved output
    return inma


def imodel_filter_scl(pscl, pars, F, F0=7.41, y0=1850):
    # this function applies a quadratic scaling to the original forcing timeseries F
    # allows for simple nonlinearities in the pattern response to forcing

    # isolate scaling parameters dictionary (pscl)
    vals = pscl.valuesdict()
    # produce quadratically scaled forcing timeseries
    fscl = vals["a"] * F + vals["b"] * np.square(F)
    # now run the convolution model with the transformed forcing timeseries
    inma = imodel_filter(pars, fscl, F0=7.41, y0=1850)
    return inma


def rmodel(eofout, us):
    # reconstruct step function output from EOFs and a user-defined PC timeseries 'us'
    # first create the synthetic EOF xarray structure
    # we copy the original EOFs and PCs from the raw data (we will keep the spatial patterns)
    eof_synth = eofout.copy()
    # now replace the PC matrix 'u' with the user defined vlaue
    eof_synth["u"] = us
    # now call recon function to reconstruct the original data from the Xarray EOF dataset
    Xrs = recon(eof_synth)
    return Xrs


def pmodel(pars, nt):
    # makes synthetic PC timeseries from parameters which define a set of exponential decay functions
    # pars is the parameter data structure
    # nt is the length of the desired output timeseries
    # nt=len(x)
    # first we make an incrementally ascending time vector 'x'
    x = np.arange(0, nt)
    # isolate the parameter dictionary
    vals = pars.valuesdict()
    # this code calculates (from the parameter names) how many EOF modes are encoded - nm
    nm = len([value for key, value in vals.items() if "c" in key.lower()])
    # this calculates (from parameter names) how many decay timeseries are encoded
    ntau = len([value for key, value in vals.items() if "t" in key.lower()])
    # print(nm)
    # intitialise the output PC timeseries with zeros
    aout = np.zeros([nt, nm])
    # now loop over the EOF modes
    for i in np.arange(0, nm):
        # first add a constant (defined per mode)
        aout[:, i] = aout[:, i] + vals["c" + str(i)]
        # now loop over the time constants represented
        for j in np.arange(0, ntau):
            # for each time constant, we add on an exponential decay function by calling expotas
            # x is the time vector
            # the second parameter S_ji is found in the dictionary - this is the coefficient
            # of this exponential decay for mode i and timescale j
            # the third parameter is the decay constant associated with the jth timescale (same for all modes)
            aout[:, i] = aout[:, i] + expotas(
                x, vals["s" + str(j) + str(i)], vals["t" + str(j)]
            )
    return aout


def residual(pars, modewgt, data=None):
    # this is used in fitting the exponential parameters used in pmodel
    # data is here a pc timeseries for the step function response (size nt by nm)
    # the function returns the weighted residual of the synthetic PC timeseries, compared with truth
    # modewgt is the weighting given to each of the modes in the PC timeseries

    # firstly, we tile the weight vector to be the same shape as data
    wgtt = np.tile(modewgt.T, (data.shape[0], 1))
    # now, we take the weighted difference between synthetic and real PCs
    return wgtt * (data - pmodel(pars, data.shape[0]))


def residual_project_scl(pscl, pars, f, modewgt, data=None):
    # this is used in fitting the quadratic forcing adjustment parameters pscl
    # data here is a PC timeseries for the convolved response/target simulation (size time by n_modes)
    # the function returns the weighted (by mode) residual of the convolved PC timeseries, compared with truth
    # modewgt is the weighting given to each of the modes in the PC timeseries

    # firstly, we tile the weight vector to be the same shape as data
    wgtt = np.tile(modewgt.T, (data.shape[0], 1))
    # fnow we call imodel_filter_scl to produce synthetic PC timeseries for the convolved data, with quadratic scaling
    mdl = imodel_filter_scl(pscl, pars, f)
    # calc difference with the data (here the data is the actual target simulation, projected onto the EOFs)
    rs = data - mdl
    # weight by mode variance
    rs = rs * wgtt
    return rs


def wgt(X):
    # returns a 1d latitude area weighting vector, given an input Xarray dataset with a lat field
    weights = np.cos(np.deg2rad(X.lat))
    weights.name = "weights"
    return weights


def wgt2(X):
    # returns a 2d latitude area weighting matrix - size n_lon by n_lat, given an input Xarray dataset with a lat field
    weights = wgt(X)
    wgt2 = np.tile(weights, (len(X.lon), 1)).T * 0.99 + 0.01
    return wgt2


def wgt3(X):
    # returns a 3d latitude area weighting matrix - size n_year by n_lon by n_lat, given an input Xarray dataset with a lat field
    weights = wgt(X)
    wgt3 = (
        np.tile(weights.T, (len(X.lon), len(X.year), 1)).transpose([1, 2, 0]) * 0.99
        + 0.01
    )
    return wgt3


def makeparams(t0, nm):
    # this creates an lmfit parameter object to define the exponential function
    # t0 is a vector of default timescales
    # this is the number of timescales in the model
    nt = len(t0)
    # initialise the parameter object
    fit_params = lmfit.Parameters()
    # loop over the timescales
    for i in np.arange(0, nt):
        # for each timescale, we add a parameter for the decay constant - default t_i
        # at the moment, we allow LMFIT 1 order magnitude limits compared with the default
        fit_params.add("t" + str(i), value=t0[i], min=t0[i] / 10, max=t0[i] * 10)
        # now loop over the number of EOF modes nm
        for j in np.arange(0, nm):
            # add a parameter representing the coefficient for the exponential decay with timescale t_i (coeff can be any value)
            fit_params.add("s" + str(i) + str(j), value=1)
    # finally, add a constant term for each parameter
    for j in np.arange(0, nm):
        fit_params.add("c" + str(j), value=0)
    return fit_params


def svds(X, nm):
    # compact wrapper for Eof - returning a dictionary with EOFS (v), PCs(u) and eigenvalues(s)
    solver = Eof(X, center=False, weights=wgt2(X))
    eofout = {}

    eofout["v"] = solver.eofs(neofs=nm, eofscaling=1)
    eofout["u"] = solver.pcs(npcs=nm, pcscaling=1)
    eofout["s"] = solver.eigenvalues(neigs=nm)
    eofout["weights"] = solver.getWeights()

    return eofout


def get_time_name(ds):
    # code to get time name variable from xarray dataset - some models are 'time', others are 'year'
    for time_name in ["time", "year"]:
        if time_name in ds.coords:
            return time_name
    raise RuntimeError("Couldn't find a time coordinate")


def get_lat_name(ds):
    # code to get lat name variable from xarray dataset
    for lat_name in ["lat", "latitude"]:
        if lat_name in ds.coords:
            return lat_name

    raise RuntimeError("Couldn't find a latitude coordinate")


def global_mean(ds):
    # returns the latitude-weighed global mean timeseries of an xarray dataset.
    # averages all dimensions other than time
    lat = ds[get_lat_name(ds)]
    tm = get_time_name(ds)
    weight = np.cos(np.deg2rad(lat))
    weight = weight / weight.mean()
    other_dims = set(ds.dims) - {tm, "ens"}
    return (ds * weight).mean(other_dims)


def get_timescales(X, t0, nm):
    # code to find optimised model parameters, given model output to a step function in an Xarray datarray X
    # t0 is the vector of timescales to be used in the fit (user choice)
    # nm is the number of modes to be used in the fit (user choice)

    # this is the length in time of the target data, assuming time is the first dimension
    nt = X.shape[0]
    # perform a PCA on the step function output, truncating to nm modes
    eofout = svds(X, nm)
    # initialise an LMFIT parameter object, allowing for the number of timescales in t0 and the number of modes nm
    fit_params = makeparams(t0, nm)
    # define a mode weighting vector, length nm (for optimization) as the root of the PCA eigenvalue
    modewgt = np.sqrt(eofout["s"])
    # Do the optimization, calling an lmfit minimization
    # residual is the function to be called and minimised in this library
    # fit_params is the LMFIT parameter initial guesses
    # residual has an argument for the weighting of each mode, we pass modewgt to that
    # the target data is the original EOF PC timeseries, u
    out = lmfit.minimize(
        residual, fit_params, args=(modewgt,), kws={"data": eofout["u"]}
    )
    # compute synthetic step-function response PCs for the model, length nt
    us = pmodel(out.params, nt)
    # convert the synthetic PCs into an xarray dataset
    usa = xr.DataArray(
        us, coords=(eofout["u"].time, eofout["u"].mode), dims=("time", "mode")
    )
    # initialise a synthetic PCA xarray structure by copying the true PCA output
    eofnew = eofout.copy()
    # substitute the original PCs with the synthetic PCs
    eofnew["u"] = usa
    # return everything
    return (out, eofout, eofnew)


def recon(eofout):
    # this code reconstructs a full dataset (time by lat by lon), given a PCA decomposition
    # input is a dictionary, containing 4 keys, u,s, v  and wgt which are the PCs, eigenvalues, EOF patterns and defined spatial weights.

    # define matrices based on dictionary input
    # u1 is the PC matrix, size n_time by n_modes
    u1 = eofout["u"]
    # s1 is the eigenvalue matrix, size n_modes
    s1 = eofout["s"]
    # v1 is the EOF patterns, size n_pixels by n_modes
    v1 = eofout["v"]
    # wgt is the area weighting matrix, size n_pixels
    wgt = eofout["weights"]
    # number of modes
    nm = v1.shape[0]
    # reshape v1 into a 2d matrix
    v1f = v1.values.reshape(nm, -1)
    # compute reconstruceted field (unweighted) as dot product
    Xr = np.dot(np.dot(u1, np.diag(s1)), v1f)
    # compute reconstruceted field (weighted) as dot product
    Xrp = np.reshape(Xr, [u1.shape[0], v1.shape[1], v1.shape[2]]) / wgt
    # convert reconstructed field to xarray and return
    Xo = xr.DataArray(
        Xrp, coords=(u1.time, v1.lat, v1.lon), dims=("time", "lat", "lon")
    )

    return Xo


# def get_patterns_pinv(X,tsp):
#     #depreciated - this does a penrose inverse to calculate the patterns associated with a basis set defined by a non-orthogonal set of exponential timeseries
#     nt=X.shape[0]
#     x_array=np.arange(1,nt+1)

#     u0=np.empty([nt, len(tsp)])
#     for i,ts in enumerate(tsp):
#         tmp=expotas(x_array,1,ts)
#         u0[:,i]=tmp/np.mean(tmp)

#     #u1[:,-1]=u0[:,-1]-np.dot(np.dot(u0[:,-1],u0[:,:-1]),np.linalg.pinv(u0[:,:-1]))
#     v1f=np.dot(np.linalg.pinv(u0),X.values.reshape(nt,-1))
#     v1=np.reshape(v1f,(len(tsp),X.shape[1],X.shape[2]))

#     #Xr=np.reshape(np.dot(u1,v1f),X.shape)
#     va=xr.DataArray(v1, coords=(np.array(tsp),X.lat,X.lon), dims=('mode','lat','lon'))

#     return va,u1


# def imodel(pars, eofout, F, F0=7.41, y0=1850):
#   #depreciated, reconstructs full grids and sums - very slow
#   nt=len(F)

#   us=pmodel(pars,nt)
#   usa=xr.DataArray(us, coords=(np.arange(y0,nt+y0),eofout['u'].mode), dims=('time','mode'))
#   Xrs=rmodel(eofout,usa)
#   inm=Xrs*0.
#   for Ft,i in enumerate(np.arange(0,nt-1)):
#     dF=(F[i+1]-F[i])/F0
#     ts=nt-i
#     inm[i:,:,:]=inm[i:,:,:]+dF*Xrs[0:ts,:,:].values
#   return inm


# def imodel_eof(pars, F, F0=7.41, y0=1850):
#   #depreciated - loop over forcing vecrtor, quite slow
#   nt=len(F)
#   vals = pars.valuesdict()
#   nm=len([value for key, value in vals.items() if 'c' in key.lower()])
#   us=pmodel(pars,nt)
#   usa=xr.DataArray(us, coords=(np.arange(y0,nt+y0),np.arange(0,nm)), dims=('time','mode'))
#   #Xrs=rmodel(eofout,usa)
#   inm=usa*0.
#   for Ft,i in enumerate(np.arange(0,nt-1)):
#     dF=(F[i+1]-F[i])/F0
#     ts=nt-i
#     inm[i:,:]=inm[i:,:]+dF*usa[0:ts,:].values
#   return inm

# def residual_project(pars, f, modewgt, data=None):
#     #this is used in fitting the exponential parameters used in pmodel
#     #the function returns the weighted residual of the synthetic PC timeseries, compared with truth
#     #modewgt is the weighting given to each of the modes in the PC timeseries

#     wgtt=np.tile(modewgt.T,(data.shape[0],1))

#     mdl=imodel_filter(pars,f )
#     rs=(data-mdl)
#     rs=rs*wgtt
#     return rs

# def gresid(pars,data=None):
#     tmp=pmodel(pars,len(data)).squeeze()
#     return tmp-data

# def global_timescales(t0,data=None):
#     pars=makeparams(t0,1)
#     tms=global_mean(data).squeeze()
#     out=lmfit.minimize(gresid, pars, kws={'data': tms})
#     ts=[]
#     for i in np.arange(0,len(t0)):
#         ts.append(out.params['t'+str(i)].value)
#     return ts

# def adjust_timescales(X,Xact,pars,t0,f):
#     #this code performs the adjustment of the fitted timescales
#     scl_params = lmfit.Parameters()
#     scl_params.add('a',value=1,min=0.7,max=1.3)
#     scl_params.add('b',value=0.001,min=0.0,max=0.5)


#     vals = pars.valuesdict()
#     nm=len([value for key, value in vals.items() if 't' in key.lower()])
#     nt=X.shape[0]
#     solver = Eof(X,center=False,weights=wgt2(X))
#     eofout=svds(X,nm)
#     modewgt=np.sqrt(eofout['s'])

#     ua=solver.projectField(Xact,neofs=nm,eofscaling=1)

#     kys=[key for key, value in vals.items()]
#     x_array=np.arange(1,nt+1)


#     out = lmfit.minimize(residual_project_scl, scl_params, args=(pars,f,modewgt,), kws={'data': ua})
#     #ts=[out.params['t1'].value,out.params['t2'].value,out.params['t3'].value]
#     return out
