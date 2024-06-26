"""
PRPATT
"""

import logging

import lmfit
import numpy as np
import pandas as pd
import xarray as xr
from scipy.linalg import pinv
from scipy.optimize import minimize

# This is a library of functions to provide the backend to the pulse-response logic in METEOR

LOGGER = logging.getLogger(__name__)


def make_anom(ds_exp, ds_cnt):
    """
    Make anomaly timeseries from experiment relative to control long term average

    Parameters
    ----------
    ds_exp : xarray.DataArray
             Dataset for experiment
    ds_cnt : xarray.DataArray
             Dataset from the control experiment

    Returns
    -------
    xarray dataset
             Dataset of anomalies
    """
    ds_anom = ds_exp - ds_cnt.mean("year", skipna=True)
    ds_anom = ds_anom.rename({"year": "time"})
    ds_anom = ds_anom.interpolate_na(dim="lat", method="nearest").interpolate_na(
        dim="lon", method="nearest"
    )
    return ds_anom


def expotas(time, coeff, decay_time):
    """
    Return single exponential pulse response function

    At time x value of exponential pulse response function with
    time scale t1 and coefficient s1

    Parameters
    ----------
    time : float
           time at which to evalute function
    coeff : float
           Coefficent of decay function
    decay_time : float
           Decay time scale

    Returns
    -------
    np.ndarray or float
         Exponential decay reps along the time array with
         coefficient coeff and decay time decay_time
    """
    return coeff * (1 - np.exp(-time / decay_time))


def imodel_filter(pars, forc_timeseries, forc_step=7.41, year_0=1850):
    """
    Convolve PC timeseries from step function response with
    forcing timeseries to produce convolved PC timeseries
    describing transient climate evolution.

    Takes forcing timeseries as input and convolves with
    synthetic PC timeseries generated from input parameters
    (derived in response to forcing step function)

    Parameters
    ----------
    pars : lmfit.parameter.Parameters
           Pattern scaling parameters
    forc_timeseries : np.ndarray
           Array of forcing timeseries to convolve with
    forc_step : float
           Size of forcing in the step function experiment
    year_0 : int
           First year of timeseries

    Returns
    -------
    xr.Dataarray
           Outputted convolution principal component timeseries
    """
    n_times = len(forc_timeseries)
    # vector of forcing differences, dF - append 0
    diff_forc = np.append(np.diff(forc_timeseries), 0) / forc_step
    # get parameter value dictionary
    vals = pars.valuesdict()
    n_modes = len([key for key in vals if "t" in key.lower()])
    # create the synthetic pulse-response kernel for a unit step function
    # output needs to be n_times in length - long enough for the first timestep of the convolution
    pc_matrix = pmodel(pars, n_times)
    # print(pc_matrix.shape)
    # make the PC timeseries into an xarray Dataarray
    pc_dataarray = xr.DataArray(
        pc_matrix,
        coords=(np.arange(year_0, n_times + year_0), np.arange(1, n_modes + 1)),
        dims=("time", "mode"),
    )
    # Convolution step - convolve the forcing difference timeseries dF with the step function kernel
    inm = np.apply_along_axis(
        lambda m: np.convolve(m, diff_forc, mode="full"), axis=0, arr=pc_dataarray
    )
    # truncate the output to the length of the forcing timeseries
    inm = inm[:n_times]
    # format the output as a datarray
    inma = xr.DataArray(
        inm,
        coords=(np.arange(year_0, n_times + year_0), np.arange(0, n_modes)),
        dims=("time", "mode"),
    )
    return inma


def rmodel(pattern_full, pc_matrix):
    """
    Reconstruct gridded, time evolving output from a user
    defined principal component timeseries and EOF patterns

    Parameters
    ----------
    pattern_full : dict
             Dictionary temporal and spatial pattern
    pc_matrix : xarray.DataArray
             Data array of principal component timeseries

    Returns
    -------
    xarray.DataArray
             Reconstructed dataarray for the forcing change
    """
    # reconstruct step function output from EOFs and a user-defined PC timeseries 'pc_matrix'
    # first create the synthetic EOF xarray structure
    # we copy the original EOFs and PCs from the raw data (we will keep the spatial patterns)
    pattern_synth = pattern_full.copy()
    # now replace the PC matrix 'u' with the user defined vlaue
    pattern_synth["u"] = pc_matrix
    # now call recon function to reconstruct the original data from the Xarray EOF dataset
    recon_data = recon(pattern_synth)
    return recon_data


def pmodel(pars, n_times):
    """
    Calculate synthetic principal component time series associated with
    a step change in forcing.  Each mode of the (n_mode) PC timeseries is constructed
    as a sum of (n_tau) exponential decay functions.


    Parameters
    ----------
    pars : lmfit.Parameters
           Object that defines an exponential decau fit
           It holds n_tau timescale decays of the n_modes,
           the n_tau by n_modes coefficents to
           these exponential decay responses and
           n_modes constant terms for the fits
    n_times : Time series length

    Returns
    -------
    np.ndarray
         pc timeseries corresponding to input exponential decay
    """
    # makes synthetic PC timeseries from parameters which define a set of exponential decay functions
    # pars is the parameter data structure
    # n_times is the length of the desired output timeseries
    # n_times=len(x)
    # first we make an incrementally ascending time vector 'x'
    time_vector = np.arange(0, n_times)
    # isolate the parameter dictionary
    vals = pars.valuesdict()
    # this calculates (from parameter names) how many decay timeseries are encoded
    ntau = len([key for key in vals if "t" in key.lower()])
    # intitialise the output PC timeseries with zeros
    aout = np.zeros([n_times, ntau])
    # now loop over the EOF modes
    for i in np.arange(0, ntau):
        # first add a constant (defined per mode)
        aout[:, i] = expotas(time_vector, vals["s" + str(i)], vals["t" + str(i)])
    return aout


def expfun(t, pars):
    """
    Calculate the sum of exponential decays
    defined by timescales and amplitudes given in pars
    at times t

    Parameters
    ----------
    t: np.array
       times for which to caluclate exonential function
    pars: lmfit.parameter.Parameters
          Timescale parameters from keys t0, s0, t1, s1 etc with timescales as
          and amplitudes as values

    Returns
    -------
    np.ndarray
           Sum of decays
    """
    vals = pars.valuesdict()
    # this calculates (from parameter names) how many decay timeseries are encoded
    ntau = len([key for key in vals if "t" in key.lower()])
    out = np.zeros(len(t))
    for i in np.arange(ntau):
        out = out + expotas(t, vals["s" + str(i)], vals["t" + str(i)])
    return out


def fit_timescales(X, a0):
    """
    Find best fit timescales for xarray X on lat lon and time

    Parameters
    ----------
    X : xr.DataArray
        Data on lat, lon and time
    a0 : np.ndarray
         of even length, containing guesses for amplitudes and
         corresponding timescales at even and odd consecutive
         placements

    Returns
    -------
    lmfit.MinimizerResult
          The result of minimizing the difference between the global
          latitudinally weighted mean timeseries of X and a sum of
          exponential decays over the time series with respect to
          the amplitudes and timescales of the decays
    """
    awgt = np.cos(X.lat / 180 * np.pi)
    awgt = awgt / np.mean(awgt)
    ts = (X * awgt).mean("lat").mean("lon").values
    fit_params = make_params(a0)
    # print(ts)
    out = lmfit.minimize(
        lambda x: np.square(ts - expfun(np.arange(0, len(ts)), x)),
        fit_params,
    )
    return out


def make_amat(pars, nt):
    """
    Make a matrix of timesteps as rows and the various exponential
    decays as columns

    Parameters
    ----------
    pars: lmfit.parameter.Parameters
          Timescale parameters from keys t0, s0, t1, s1 etc with timescales as
          t0, t1 etc and amplitudes s0, s1 etc...
    nt : int
         Number of timesteps

    Returns
    -------
    np.ndarray
         Matrix with the exponential decay value at the nt timesteps
        decays along each row for each exponential decay column
    """
    vals = pars.valuesdict()
    na = len([key for key in vals if "t" in key.lower()])
    amat = np.zeros([nt, na])
    t = np.arange(nt)
    for i in np.arange(na):
        amat[:, i] = expotas(t, vals["s" + str(i)], vals["t" + str(i)])
    return amat


def make_pmat(tauvec, nt):
    """
    Make a matrix of timesteps as columns and exponential
    decays as rows

    Parameters
    ----------
    nt : int
         Number of timesteps
    tauvec : np.ndarray
        Vector of timescales

    Returns
    -------
    np.ndarray
         Matrix with the exponential decay value at the nt timesteps
        decays along each column for each exponential decay row
    """
    # TODO: Combine this with make_amat
    out = np.zeros((len(tauvec), nt))
    for i, t in enumerate(tauvec):
        out[i, :] = expotas(np.arange(nt), 1, t)
    return out


def proj_gm(tauvec, gmanom, fcg_aer):
    """
    Make projection of the global mean anomaly onto timevectors of
    the exponential responses convolved with the derivative of the
    aerosol forcing timeseries

    Leveraging the make_pmat to make a matix of exponential responses
    for the tauvec and convolving that with the derivative of the
    aerosol forcing, we the get a timevector for the forcing decay (T).
    We the project that onto the global mean anomaly of the time series
    (G) as T^-1 G T

    Parameters
    ----------
    tauvec : np.ndarray
        Vector of timescales
    gmanom : xarray.DataArray
        Global mean of residual between result from other forcers
        and input data
    fcg_aer : xarray.DataArray
        Timeseries of aerosol forcing

    Returns
    -------
    np.ndarray
        Projection of the global mean anomaly onto timevectors of
        the exponential responses convolved with the derivative
        of the aerosol forcing timeseries
    """
    # TODO: 500 or len(gmanom['time'])? Think 500 is actually alright possibly max
    # TODO: Timediff now assumes Gregorian calendar, maybe allow for others?
    pmat = make_pmat(tauvec, 500)
    timediff = round(
        pd.Timedelta(gmanom["time"][0].values - fcg_aer["time"][0].values)
        / pd.Timedelta("365.2425 days")
    )
    print(timediff)
    timvec = np.apply_along_axis(
        lambda m: np.convolve(m, fcg_aer.diff("time"), mode="full"), axis=1, arr=pmat
    )[:, timediff : len(fcg_aer) + timediff]
    print(timvec.shape)
    print(len(gmanom["time"]))
    print(gmanom.shape)
    projvec = np.dot(
        np.dot(gmanom[:], pinv(timvec[:, : len(gmanom["time"])])),
        timvec[:, : len(gmanom["time"])],
    )
    return projvec


def rmse_gm(tauvec, gmanom, fcg_aer):
    """
    Calculate root mean square error between projection of
    global mean time series as described above and global
    mean timeseries

    Parameters
    ----------
    tauvec : np.ndarray
        Vector of timescales
    gmanom : xarray.DataArray
        Global mean of residual between result from other forcers
        and input data
    fcg_aer : xarray.DataArray
        Timeseries of aerosol forcing

    Returns
    -------
    float
        Root mean square error
    """
    projvec = proj_gm(tauvec, gmanom, fcg_aer)
    err = projvec - gmanom
    return np.sum(err**2)


def residual(pars, modewgt, data):
    """
    Calculate weighted residual between the step function response
    PC timeseries and the reconstruction of those PC timeseries using
    the pmodel function.   The contribution of each mode in the combined
    residual is weighted by a vector modewgt.

    Parameters
    ----------
    pars : lmfit.parameter.Parameters
           Pattern scaling parameters
    modewgt : np.ndarray
           Principal component weights
    data : xr.DataArray
           PC timeseries for convolved response/target simulation
    Returns
    -------
        xr.DataArray
           Weighted residual between model and synthetic prediction fit
    """
    # this is used in fitting the exponential parameters used in pmodel
    # data is here a pc timeseries for the step function response (size n_times by n_modes)
    # the function returns the weighted residual of the synthetic PC timeseries, compared with truth
    # modewgt is the weighting given to each of the modes in the PC timeseries

    # firstly, we tile the weight vector to be the same shape as data
    wgtt = np.tile(modewgt.T, (data.shape[0], 1))
    # now, we take the weighted difference between synthetic and real PCs
    return wgtt * (data - pmodel(pars, data.shape[0]))


def wgt(array_w_lat):
    """
    Calculate cosine weights for an xarray with latitude field

    Parameters
    ----------
    array_w_lat: xarray.DataArray
                 Array that has latitudinal dimension
    Returns
    -------
    xarray.Datarray
                 1d xarray called weights, with weights per latitude
    """
    weights = np.cos(np.deg2rad(array_w_lat.lat))
    weights.name = "weights"
    return weights


def wgt2(array_w_latlon):
    """
    Calculate cosine weights for an xarray with latitude field on 2d

    Parameters
    ----------
    array_w_latlon: xarray.DataArray
                 Array that has latitudinal and longitudinal
                 dimension
    Returns
    -------
    xarray.Datarray
                 2d xarray called weights, with weights per latitude
                 on logitude by latitude grid
    """
    weights = wgt(array_w_latlon)
    weights_2d = np.tile(weights, (len(array_w_latlon.lon), 1)).T * 0.99 + 0.01
    return weights_2d


def wgt3(array_w_latlontime):
    """
    Calculate cosine weights for an xarray with latitude field on 2d

    Parameters
    ----------
    array_w_latlontime: xarray.DataArray
                  Array that has latitudinal and longitudinal
                  dimension

    Returns
    -------
    xarray.Datarray
                  2d xarray called weights, with weights per latitude
                  on logitude by latitude grid
    """
    weights = wgt(array_w_latlontime)
    weights_3d = (
        np.tile(
            weights.T, (len(array_w_latlontime.lon), len(array_w_latlontime.year), 1)
        ).transpose([1, 2, 0])
        * 0.99
        + 0.01
    )
    return weights_3d


def make_params(ain):
    """
    Create lmfit parameter object to define parameters used by pmodel to create
    synthetic step function response PCs.

    Parameters
    ----------
    ain : list
          Of even length, initial guess of amplitudes and timescales
    Returns
    -------
    lmfit.Parameters
          Object that can accommodate n_times timescale decays of the n_modes
          where both the timescales, the n_times by n_modes coefficents to
          these exponential responses and n_modes constant terms for the fits
    """
    # this creates an lmfit parameter object to define the exponential function
    # t0 is a vector of default timescales
    # this is the number of timescales in the model
    n_times = int(len(ain) / 2)
    # initialise the parameter object
    fit_params = lmfit.Parameters()
    # loop over the timescales
    for i in np.arange(0, n_times):
        # for each timescale, we add a parameter for the decay constant - default t_i
        # at the moment, we allow LMFIT 1 order magnitude limits compared with the default
        fit_params.add(
            "t" + str(i), value=ain[2 * i + 1], min=0, max=ain[2 * i + 1] * 10
        )

        # add a parameter representing the coefficient for the exponential decay with timescale t_i (coeff can be any value)
        fit_params.add("s" + str(i), value=ain[2 * i])

    return fit_params


def get_time_name(ds):
    """
    Get name of temporal dimension

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    str
        The name of the temporal dimension of the dataset,
        provided it's either time or year

    Raises
    ------
    RuntimeError
        If there is no dimension called time or year
        in the dataset
    """
    for time_name in ["time", "year"]:
        if time_name in ds.coords:
            return time_name
    raise RuntimeError("Couldn't find a time coordinate")


def get_lat_name(ds):
    """
    Get name of latitude dimension

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    str
        The name of the latitudinal dimension of the dataset,
        provided it's either lat or latitude

    Raises
    ------
    RuntimeError
        If there is no dimension called lat or latitude
        in the dataset
    """
    for lat_name in ["lat", "latitude"]:
        if lat_name in ds.coords:
            return lat_name

    raise RuntimeError("Couldn't find a latitude coordinate")


def global_mean(ds):
    """
    Calculate latitude weighted global mean of xarray dataset

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    xarray.DataArray
          Global mean of the dataset over dimensions that are not
          the time dimension, or the dimension "ens"
    """
    lat = ds[get_lat_name(ds)]
    time_dim_name = get_time_name(ds)
    weight = np.cos(np.deg2rad(lat))
    weight = weight / weight.mean()
    other_dims = set(ds.dims) - {time_dim_name, "ens"}
    return (ds * weight).mean(other_dims, skipna=True)


def get_timescales(anomaly_data, n_modes):
    """
    Calculate optimised parameters by minimising the residual
    between the output of pmodel and the step function PC timeseries
    represented in anomaly_data, given initial guesses on the timescales
    present in the ouput (tmscl_0) and the number of modes retained
    in the PCA (n_modes)


    Find optimised model parameters given model anomaly matrix from
    step function forcing experiment

    Parameters
    ----------
    anomaly_data : xarray.DataArray
                   Data array with the anomaly between the experiment
                   and control experriment. Assumed to have time
                   as first dimension
    n_modes : int
              Number of modes to include

    Returns
    -------
    list
        Elements are: out - the result of the lmfit fitting,
        pattern - a dictionary containing the pattern in terms of
        the temporal part, u, which has a timeseries per mode, and
        a spatial part, v, which has a spatial pattern for each model
    """
    # initialise an LMFIT parameter object, with tmscl_0 timescales and
    # n_modes modes
    ampguess = anomaly_data.mean("lat").mean("lon").mean("time")

    tguess = 10 ** (np.arange(n_modes) + 1)

    a0 = np.zeros(n_modes * 2)
    for i, t in enumerate(tguess):
        a0[2 * i] = ampguess
        a0[2 * i + 1] = t

    aopt = fit_timescales(anomaly_data, a0)
    pattern = {}
    print(aopt)
    print(aopt.params)
    # sys.exit(4)
    u_np = make_amat(aopt.params, len(anomaly_data.time))
    uxr = xr.DataArray(
        data=u_np,
        dims=["time", "mode"],
        coords={
            "time": (["time"], anomaly_data.time.data),
            "mode": (["mode"], np.arange(n_modes)),
        },
    )
    pattern["u"] = uxr

    ui = pinv(u_np)
    b = np.tensordot(ui, anomaly_data.values, axes=1)
    bx = xr.DataArray(
        data=b,
        dims=["mode", "lat", "lon"],
        coords={
            "lat": (["lat"], anomaly_data.lat.data),
            "lon": (["lon"], anomaly_data.lon.data),
            "mode": (["mode"], np.arange(n_modes)),
        },
    )
    pattern["v"] = bx
    # return everything
    return (aopt, pattern)


def get_timescales_from_anomaly(residual_anom, fcg_aer, n_modes=2):
    """
    Calculate optimised parameters by minimising the residual
    between the output of pmodel and the step function PC timeseries
    represented in anomaly_data, given initial guesses on the timescales
    present in the ouput (tmscl_0) and the number of modes retained
    in the PCA (n_modes)


    Find optimised model parameters given model anomaly matrix from
    step function forcing experiment

    Parameters
    ----------
    residual_anom : xarray.DataArray
        Data array with the residual between the experiment
        predicted from abrupt change experiments and anomaly
        experiment. Assumed to have time as first dimension
    fcg_aer : np.ndarray
        Timeseries of aerosol forcing
    n_modes : Number of modes to fit

    Returns
    -------
    list
        Elements are: out - the result of the lmfit fitting,
        pattern - a dictionary containing the pattern in terms of
        the temporal part, u, which has a timeseries per mode, and
        a spatial part, v, which has a spatial pattern for each model
    """
    if n_modes != 2:
        LOGGER.warning(
            "n_modes different from 2 is currently not supported for residual"
        )
        n_modes = 2
    nt = len(residual_anom.time)
    pattern = {}
    gmanom = global_mean(residual_anom)
    # TODO: Are all amplitudes = 1 a valid assumption?
    # TODO: Modify this to fit to nmodes different from 2?
    bounds = [(1, 10), (10, 100)]
    opt = minimize(
        rmse_gm, [5, 50], args=(gmanom, fcg_aer), bounds=(bounds[0], bounds[1])
    )
    params = lmfit.Parameters()

    for i in range(n_modes):
        params.add(f"t{i}", value=opt.x[i], min=bounds[i][0], max=bounds[i][1])
        params.add(f"s{i}", value=1, min=0.9, max=1.1)

    # make exponential decay timeseries with the optimized time constants
    # TODO: Check: nt here used to be 500, but I think this is more correct
    pmat = make_pmat(opt.x, nt)

    uxr = xr.DataArray(
        data=pmat.T,
        dims=["time", "mode"],
        coords={
            "time": (["time"], residual_anom.time.data),
            "mode": (["mode"], np.arange(n_modes)),
        },
    )
    pattern["u"] = uxr
    # convolve the aerosol forcing difference timeseries with the timeseries
    timvec = np.apply_along_axis(
        lambda m: np.convolve(m, fcg_aer.diff("time"), mode="full"), axis=1, arr=pmat
    )[:, 100 : len(fcg_aer) + 100]
    # invert time convolution of the aerosol-pulse response matrix
    # project the time inverse matrix onto the aerosol anomaly map,
    # and convert to xarray to get the spatial patterns associated with each decay mode
    tmp = np.tensordot(pinv(timvec[:, :nt]), residual_anom[:, :, :], (0, 0))
    bx = xr.DataArray(
        data=tmp,
        dims=["mode", "lat", "lon"],
        coords={
            "lat": (["lat"], residual_anom.lat.data),
            "lon": (["lon"], residual_anom.lon.data),
            "mode": (["mode"], np.arange(n_modes)),
        },
    )
    pattern["v"] = bx
    return (params, pattern)


def recon(eofout):
    """
    Reconstruct full dataset given a PCA decompostion represented
    in the dictionary format outputted by eof_calculation_wrapper

    Output in xarray dataarray format (time by lat by lon)

    Parameters
    ----------
    eofout : dict
             Dictionary, containing 2 keys, u and v
             which are timeseries for the pulse response,
             per mode and the corresponding
             spatial patterns.

    Returns
    -------
    xarray.DataArray
           Reconstructed field in space and time as xarray
    """
    # Define matrices based on dictionary input:
    mode_timescales = eofout["u"]  # size n_time by n_modes
    pattern_per_mode = eofout["v"]  # size n_pixels by n_modes
    # number of modes
    n_modes = pattern_per_mode.shape[0]
    # reshape v1 into a 2d matrix
    pattern_2d = pattern_per_mode.values.reshape(n_modes, -1)
    # compute reconstruceted field (unweighted) as dot product
    recon_unweighted = np.dot(mode_timescales, pattern_2d)
    # compute reconstruceted field (weighted) as dot product
    recon_weighted = np.reshape(
        recon_unweighted,
        [
            mode_timescales.shape[0],
            pattern_per_mode.shape[1],
            pattern_per_mode.shape[2],
        ],
    )
    # convert reconstructed field to xarray and return
    recon_xarray = xr.DataArray(
        recon_weighted,
        coords=(mode_timescales.time, pattern_per_mode.lat, pattern_per_mode.lon),
        dims=("time", "lat", "lon"),
    )
    return recon_xarray


# def imodel_filter_scl(pscl, pars, F, F0=7.41, y0=1850):
#     """
#     Apply a quadratic scaling to forcing before convolving

#     Apply a quadratic scaling to the original forcing timeseries
#     before convolving to make predictions. Allows for simple
#     nonlinearities in the pattern response to forcing.


#     Parameters
#     ----------
#     pscl : lmfit.parameter.Parameters
#            Quadratic fit scale parameters
#     pars : lmfit.parameter.Parameters
#            Pattern scaling parameters
#     F : np.ndarray
#            Array of forcing timeseries to convolve with
#     F0 : float
#            Size of forcing in the step function experiment
#     y0 : int
#            First year of timeseries

#     Returns
#     -------
#     xr.Dataarray
#            Outputted convolution principal component timeseries
#     """
#     # isolate scaling parameters dictionary (pscl)
#     print(type(pscl))
#     vals = pscl.valuesdict()
#     # produce quadratically scaled forcing timeseries
#     fscl = vals["a"] * F + vals["b"] * np.square(F)
#     # now run the convolution model with the transformed forcing timeseries
#     inma = imodel_filter(pars, fscl, F0=F0, y0=y0)
#     return inma

# def residual_project_scl(pscl, pars, forc_timeseries, modewgt, data):
#     """
#     Get residual from quadratic forcing adjustment fit
#     !NB: Ben please check this extra cearfully, not at all sure I understood what this does...

#     Parameters
#     ----------
#     pscl : lmfit.parameter.Parameters
#            Quadratic fit scale parameters
#     pars : lmfit.parameter.Parameters
#            Pattern scaling parameters
#     forc_timseries : np.ndarray
#            Array of forcing timeseries to convolve with
#     modewgt : np.ndarray
#            Principal component weights
#     data : xr.DataArray
#            PC timeseries for convolved response/target simulation

#     Returns
#     -------
#         xr.DataArray
#            Weighted residual between model and synthetic prediction fit
#     """
#     # this is used in fitting the quadratic forcing adjustment parameters pscl
#     # data here is a PC timeseries for the convolved response/target simulation (size time by n_modes)
#     # the function returns the weighted (by mode) residual of the convolved PC timeseries, compared with truth
#     # modewgt is the weighting given to each of the modes in the PC timeseries

#     # firstly, we tile the weight vector to be the same shape as data
#     wgtt = np.tile(modewgt.T, (data.shape[0], 1))
#     # fnow we call imodel_filter_scl to produce synthetic PC timeseries for the convolved data, with quadratic scaling
#     mdl = imodel_filter_scl(pscl, pars, forc_timeseries)
#     # calc difference with the data (here the data is the actual target simulation, projected onto the EOFs)
#     residual = data - mdl
#     # weight by mode variance
#     residual = residual * wgtt
#     return residual

# def get_patterns_pinv(X,tsp):
#     #depreciated - this does a penrose inverse to calculate the patterns associated with a basis set defined by a non-orthogonal set of exponential timeseries
#     n_times=X.shape[0]
#     x_array=np.arange(1,n_times+1)

#     u0=np.empty([n_times, len(tsp)])
#     for i,ts in enumerate(tsp):
#         tmp=expotas(x_array,1,ts)
#         u0[:,i]=tmp/np.mean(tmp)

#     #u1[:,-1]=u0[:,-1]-np.dot(np.dot(u0[:,-1],u0[:,:-1]),np.linalg.pinv(u0[:,:-1]))
#     v1f=np.dot(np.linalg.pinv(u0),X.values.reshape(n_times,-1))
#     v1=np.reshape(v1f,(len(tsp),X.shape[1],X.shape[2]))

#     #Xr=np.reshape(np.dot(u1,v1f),X.shape)
#     va=xr.DataArray(v1, coords=(np.array(tsp),X.lat,X.lon), dims=('mode','lat','lon'))

#     return va,u1


# def imodel(pars, eofout, F, F0=7.41, y0=1850):
#   #depreciated, reconstructs full grids and sums - very slow
#   n_times=len(F)

#   pc_matrix=pmodel(pars,n_times)
#   usa=xr.DataArray(pc_matrix, coords=(np.arange(y0,n_times+y0),eofout['u'].mode), dims=('time','mode'))
#   Xrs=rmodel(eofout,usa)
#   inm=Xrs*0.
#   for Ft,i in enumerate(np.arange(0,n_times-1)):
#     dF=(F[i+1]-F[i])/F0
#     ts=n_times-i
#     inm[i:,:,:]=inm[i:,:,:]+dF*Xrs[0:ts,:,:].values
#   return inm


# def imodel_eof(pars, F, F0=7.41, y0=1850):
#   #depreciated - loop over forcing vecrtor, quite slow
#   n_times=len(F)
#   vals = pars.valuesdict()
#   n_modes=len([value for key, value in vals.items() if 'c' in key.lower()])
#   pc_matrix=pmodel(pars,n_times)
#   usa=xr.DataArray(pc_matrix, coords=(np.arange(y0,n_times+y0),np.arange(0,n_modes)), dims=('time','mode'))
#   #Xrs=rmodel(eofout,usa)
#   inm=usa*0.
#   for Ft,i in enumerate(np.arange(0,n_times-1)):
#     dF=(F[i+1]-F[i])/F0
#     ts=n_times-i
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
#     pars=make_params(t0,1)
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
#     n_modes=len([value for key, value in vals.items() if 't' in key.lower()])
#     n_times=X.shape[0]
#     solver = Eof(X,center=False,weights=wgt2(X))
#     eofout=eof_calculation_wrapper(X,n_modes)
#     modewgt=np.sqrt(eofout['s'])

#     ua=solver.projectField(Xact,neofs=n_modes,eofscaling=1)

#     kys=[key for key, value in vals.items()]
#     x_array=np.arange(1,n_times+1)


#     out = lmfit.minimize(residual_project_scl, scl_params, args=(pars,f,modewgt,), kws={'data': ua})
#     #ts=[out.params['t1'].value,out.params['t2'].value,out.params['t3'].value]
#     return out
