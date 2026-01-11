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
    ts = (X * awgt).mean("lat", skipna=True).mean("lon", skipna=True).values
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
    aerosol forcing, we then get a timevector for the forcing decay (T).
    We then project that onto the global mean anomaly of the time series
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
    timvec = np.apply_along_axis(
        lambda m: np.convolve(m, fcg_aer.diff("time"), mode="full"), axis=1, arr=pmat
    )[:, timediff : len(fcg_aer) + timediff]
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
            "t" + str(i),
            value=ain[2 * i + 1],
            min=ain[2 * i + 1] / 5,
            max=ain[2 * i + 1] * 2,
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
    for time_name in ["time", "year", "month"]:
        if time_name in ds.coords:
            return time_name
    raise RuntimeError("Couldn't find a time coordinate")


def get_lat_name(ds):
    """
    Get name of latitude dimension

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray

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
    # Common latitude coordinate names
    lat_variants = ["lat", "latitude", "y", "lat_rho"]

    for lat_name in lat_variants:
        if lat_name in ds.coords or lat_name in ds.dims:
            return lat_name

    raise RuntimeError(f"Couldn't find a latitude coordinate. Tried: {lat_variants}")


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-branches
def global_mean(
    ds,
    lat_name=None,
    lon_name=None,
    weights=None,
    normalize_weights=True,
    skip_dims=None,
):
    """
    Calculate latitude weighted global mean of xarray dataset or dataarray.

    This is the universal global mean function for METEOR. It can handle both
    Datasets and DataArrays with flexible coordinate detection and weighting.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Data to compute global mean over
    lat_name : str, optional
        Name of latitude coordinate (auto-detected if None)
    lon_name : str, optional
        Name of longitude coordinate (auto-detected if None)
    weights : xarray.DataArray, optional
        Custom weights for averaging (if None, uses cosine of latitude)
    normalize_weights : bool, optional
        Whether to normalize weights by their mean (default True for backward compatibility)
    skip_dims : list of str, optional
        Dimensions to skip when averaging (if None, skips 'time' and 'ens' for backward compatibility)

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Global mean with spatial dimensions removed
    """
    # Auto-detect latitude coordinate if not provided
    if lat_name is None:
        lat_name = get_lat_name(ds)

    # Auto-detect longitude coordinate if not provided
    if lon_name is None:
        lon_variants = ["lon", "longitude", "x", "lon_rho"]
        for lon_var in lon_variants:
            if lon_var in ds.coords or lon_var in ds.dims:
                lon_name = lon_var
                break
        if lon_name is None:
            raise RuntimeError(
                f"Couldn't find a longitude coordinate. Tried: {lon_variants}"
            )

    # Set default skip dimensions for backward compatibility
    if skip_dims is None:
        skip_dims = ["time", "ens"]
        # Add time dimension name detection for more robust backward compatibility
        if hasattr(ds, "dims"):
            time_variants = ["time", "month", "year"]
            for time_var in time_variants:
                if time_var in ds.dims and time_var not in skip_dims:
                    skip_dims.append(time_var)

    # Get spatial dimensions to average over
    if hasattr(ds, "dims"):
        spatial_dims = [dim for dim in ds.dims if dim not in skip_dims]
        # Ensure lat/lon are in spatial dims if they exist
        if lat_name in ds.dims and lat_name not in spatial_dims:
            spatial_dims.append(lat_name)
        if lon_name in ds.dims and lon_name not in spatial_dims:
            spatial_dims.append(lon_name)
    else:
        spatial_dims = [lat_name, lon_name]

    # Create weights if not provided
    if weights is None:
        lat = ds[lat_name]
        weights = np.cos(np.deg2rad(lat))

        # Broadcast to longitude if needed
        if lon_name in ds.dims:
            weights = weights * xr.ones_like(ds[lon_name])

    # Normalize weights if requested (default behavior for backward compatibility)
    if normalize_weights:
        weights = weights / weights.mean()

    # Calculate weighted mean
    weighted_data = ds * weights
    result = weighted_data.mean(spatial_dims, skipna=True)

    # Update attributes if possible
    if hasattr(result, "attrs"):
        if hasattr(ds, "attrs"):
            result.attrs.update(ds.attrs)
        result.attrs["operation"] = "area_weighted_global_mean"

    return result


def create_region_mask(ds, bbox=None, mask=None):
    """
    Create a 2D boolean mask for a custom region.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Data with lat/lon coordinates
    bbox : dict, optional
        Bounding box: {'lat': (min, max), 'lon': (min, max)}
    mask : numpy.ndarray or xarray.DataArray, optional
        Pre-computed boolean mask

    Returns
    -------
    xarray.DataArray
        2D boolean mask (True = inside region)
    """
    if bbox is not None:
        # Create mask from bounding box
        lat_name = get_lat_name(ds)

        lon_variants = ["lon", "longitude", "x", "lon_rho"]
        lon_name = None
        for lon_var in lon_variants:
            if lon_var in ds.coords or lon_var in ds.dims:
                lon_name = lon_var
                break
        if lon_name is None:
            raise RuntimeError("Couldn't find longitude coordinate")

        lat = ds[lat_name]
        lon = ds[lon_name]

        lat_min, lat_max = bbox["lat"]
        lon_min, lon_max = bbox["lon"]

        # Create boolean mask for latitude
        mask_lat = (lat >= lat_min) & (lat <= lat_max)

        # Handle longitude wrapping (for regions crossing 0° or 180°)
        if lon_min > lon_max:
            # Region crosses the prime meridian (e.g., -10° to 10° stored as 350° to 10°)
            mask_lon = (lon >= lon_min) | (lon <= lon_max)
        else:
            # Normal case: region doesn't wrap
            mask_lon = (lon >= lon_min) & (lon <= lon_max)

        # Combine masks
        region_mask = mask_lat & mask_lon
        return region_mask

    elif mask is not None:
        # Use provided mask
        if isinstance(mask, np.ndarray):
            # Convert to xarray with appropriate coordinates
            lat_name = get_lat_name(ds)
            lon_variants = ["lon", "longitude", "x", "lon_rho"]
            lon_name = None
            for lon_var in lon_variants:
                if lon_var in ds.coords or lon_var in ds.dims:
                    lon_name = lon_var
                    break

            coords = {lat_name: ds[lat_name], lon_name: ds[lon_name]}
            mask = xr.DataArray(mask, coords=coords, dims=[lat_name, lon_name])
        return mask

    else:
        raise ValueError("Must provide either bbox or mask")


def regional_mean(
    ds,
    region_code=None,
    region_mask=None,
    lat_name=None,
    lon_name=None,
    weights=None,
    normalize_weights=True,
):
    """
    Calculate area-weighted regional mean for a specific AR6 region or custom region.

    This function uses the IPCC AR6 reference regions from regionmask to
    calculate spatial averages over specific regions (e.g., 'EAS' for East Asia),
    or accepts a custom boolean mask for arbitrary regions.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Data to compute regional mean over
    region_code : str, optional
        AR6 region abbreviation (e.g., 'WNA', 'NEU', 'EAS', 'ARP', etc.)
        Use list_ar6_regions() to see all available regions
    region_mask : xarray.DataArray, optional
        Custom 2D boolean mask (True = inside region)
    lat_name : str, optional
        Name of latitude coordinate (auto-detected if None)
    lon_name : str, optional
        Name of longitude coordinate (auto-detected if None)
    weights : xarray.DataArray, optional
        Custom weights for averaging (if None, uses cosine of latitude)
    normalize_weights : bool, optional
        Whether to normalize weights by their mean (default True)

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Regional mean with spatial dimensions removed
        Includes 'region_code' and 'region_name' in attributes

    Examples
    --------
    >>> # Calculate mean temperature over East Asia
    >>> tas_eas = regional_mean(temperature_data, region_code='EAS')
    >>>
    >>> # Calculate mean precipitation over custom region
    >>> custom_mask = create_region_mask(data, bbox={'lat': (45, 55), 'lon': (5, 20)})
    >>> pr_custom = regional_mean(precipitation_data, region_mask=custom_mask)

    See Also
    --------
    create_region_mask : Create custom region masks
    list_ar6_regions : List all available AR6 regions
    global_mean : Calculate global mean
    extract_point : Extract time series at a specific point
    """
    # Handle custom region mask first
    if region_mask is not None:
        # Auto-detect coordinates
        if lat_name is None:
            lat_name = get_lat_name(ds)
        if lon_name is None:
            lon_variants = ["lon", "longitude", "x", "lon_rho"]
            for lon_var in lon_variants:
                if lon_var in ds.coords or lon_var in ds.dims:
                    lon_name = lon_var
                    break
            if lon_name is None:
                raise RuntimeError("Couldn't find longitude coordinate")

        # Apply mask
        regional_data = ds.where(region_mask)

        # Create weights
        if weights is None:
            lat = regional_data[lat_name]
            weights = np.cos(np.deg2rad(lat))
            if lon_name in regional_data.dims:
                weights = weights * xr.ones_like(regional_data[lon_name])

        # Apply mask to weights
        weights = weights.where(region_mask)

        # Normalize weights
        if normalize_weights:
            weights = weights / weights.mean()

        # Calculate weighted mean
        spatial_dims = [lat_name, lon_name]
        weighted_data = regional_data * weights
        result = weighted_data.mean(spatial_dims, skipna=True)

        # Update attributes
        if hasattr(result, "attrs"):
            if hasattr(ds, "attrs"):
                result.attrs.update(ds.attrs)
            result.attrs["operation"] = "area_weighted_regional_mean"
            result.attrs["region_type"] = "custom"

        return result

    # Handle AR6 region code
    if region_code is None:
        raise ValueError("Must provide either region_code or region_mask")

    try:
        import regionmask
    except ImportError:
        raise ImportError(
            "regionmask is required for AR6 regions. "
            "Install it with: pip install regionmask"
        )

    # Auto-detect latitude coordinate if not provided
    if lat_name is None:
        lat_name = get_lat_name(ds)

    # Auto-detect longitude coordinate if not provided
    if lon_name is None:
        lon_variants = ["lon", "longitude", "x", "lon_rho"]
        for lon_var in lon_variants:
            if lon_var in ds.coords or lon_var in ds.dims:
                lon_name = lon_var
                break
        if lon_name is None:
            raise RuntimeError(
                f"Couldn't find a longitude coordinate. Tried: {lon_variants}"
            )

    # Load AR6 regions
    ar6_regions = regionmask.defined_regions.ar6.all

    # Create mask for the data grid
    mask = ar6_regions.mask(ds)

    # Find the region number for the given code
    region_number = None
    region_name = None
    for region in ar6_regions:
        if region.abbrev == region_code:
            region_number = region.number
            region_name = region.name
            break

    if region_number is None:
        raise ValueError(
            f"Region code '{region_code}' not found in AR6 regions. "
            f"Use list_ar6_regions() to see available regions."
        )

    # Apply mask to data (keep only the specified region)
    regional_data = ds.where(mask == region_number)

    # Create weights if not provided
    if weights is None:
        lat = regional_data[lat_name]
        weights = np.cos(np.deg2rad(lat))

        # Broadcast to longitude if needed
        if lon_name in regional_data.dims:
            weights = weights * xr.ones_like(regional_data[lon_name])

    # Apply mask to weights as well
    weights = weights.where(mask == region_number)

    # Normalize weights if requested
    if normalize_weights:
        weights = weights / weights.mean()

    # Get spatial dimensions to average over
    spatial_dims = [lat_name, lon_name]

    # Calculate weighted mean
    weighted_data = regional_data * weights
    result = weighted_data.mean(spatial_dims, skipna=True)

    # Update attributes
    if hasattr(result, "attrs"):
        if hasattr(ds, "attrs"):
            result.attrs.update(ds.attrs)
        result.attrs["operation"] = "area_weighted_regional_mean"
        result.attrs["region_code"] = region_code
        result.attrs["region_name"] = region_name

    return result


def extract_point(
    ds,
    lat_point,
    lon_point,
    method="nearest",
    lat_name=None,
    lon_name=None,
):
    """
    Extract time series at a specific latitude/longitude point.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Data to extract point from
    lat_point : float
        Latitude of the point (degrees North, -90 to 90)
    lon_point : float
        Longitude of the point (degrees East, -180 to 180 or 0 to 360)
    method : str, optional
        Selection method: 'nearest' (default) or 'interp' for interpolation
    lat_name : str, optional
        Name of latitude coordinate (auto-detected if None)
    lon_name : str, optional
        Name of longitude coordinate (auto-detected if None)

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Time series at the specified point
        Includes 'lat_point' and 'lon_point' in attributes

    Examples
    --------
    >>> # Extract temperature at Mumbai (19.08°N, 72.88°E)
    >>> tas_mumbai = extract_point(temperature_data, lat_point=19.08, lon_point=72.88)
    >>>
    >>> # Extract with interpolation instead of nearest neighbor
    >>> tas_mumbai_interp = extract_point(temperature_data, 19.08, 72.88, method='interp')

    See Also
    --------
    regional_mean : Calculate regional mean
    global_mean : Calculate global mean
    """
    # Auto-detect latitude coordinate if not provided
    if lat_name is None:
        lat_name = get_lat_name(ds)

    # Auto-detect longitude coordinate if not provided
    if lon_name is None:
        lon_variants = ["lon", "longitude", "x", "lon_rho"]
        for lon_var in lon_variants:
            if lon_var in ds.coords or lon_var in ds.dims:
                lon_name = lon_var
                break
        if lon_name is None:
            raise RuntimeError(
                f"Couldn't find a longitude coordinate. Tried: {lon_variants}"
            )

    # Handle longitude wrapping (convert -180:180 to 0:360 if needed)
    lon_data = ds[lon_name].values
    if lon_data.max() > 180 and lon_point < 0:
        lon_point = lon_point + 360
    elif lon_data.max() <= 180 and lon_point > 180:
        lon_point = lon_point - 360

    # Extract point based on method
    if method == "nearest":
        result = ds.sel({lat_name: lat_point, lon_name: lon_point}, method="nearest")
    elif method == "interp":
        result = ds.interp({lat_name: lat_point, lon_name: lon_point})
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'nearest' or 'interp'.")

    # Update attributes
    if hasattr(result, "attrs"):
        if hasattr(ds, "attrs"):
            result.attrs.update(ds.attrs)
        result.attrs["operation"] = f"point_extraction_{method}"
        result.attrs["lat_point"] = lat_point
        result.attrs["lon_point"] = lon_point

    return result


def list_ar6_regions():
    """
    List all available IPCC AR6 reference regions.

    Prints a formatted table of region codes and names that can be used
    with the regional_mean() function.

    Returns
    -------
    None
        Prints region information to stdout

    Examples
    --------
    >>> list_ar6_regions()
    AR6 IPCC Reference Regions:
    ============================================================
      GIC    : Greenland/Iceland
      NWN    : N.W. North America
      NEN    : N.E. North America
      ...

    See Also
    --------
    regional_mean : Calculate regional mean using AR6 regions
    """
    try:
        import regionmask
    except ImportError:
        raise ImportError(
            "regionmask is required for list_ar6_regions(). "
            "Install it with: pip install regionmask"
        )

    ar6_regions = regionmask.defined_regions.ar6.all

    print("AR6 IPCC Reference Regions:")
    print("=" * 60)
    for region in ar6_regions:
        print(f"  {region.abbrev:6s} : {region.name}")
    print("=" * 60)
    print(f"Total: {len(ar6_regions)} regions")


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
    # our first guess for amplitude is just the mean of the 2d field
    ampguess = (
        anomaly_data.mean("lat", skipna=True)
        .mean("lon", skipna=True)
        .mean("time", skipna=True)
    )
    # our time guess is 1,10,100 etc years
    tguess = 5 * 10 ** (np.arange(n_modes))
    # initialise the initial guess vector
    a0 = np.zeros(n_modes * 2)
    for i, t in enumerate(tguess):
        a0[2 * i] = ampguess
        a0[2 * i + 1] = t
    # fit the timescales using lmfit to fit global mean of the anomaly data
    aopt = fit_timescales(anomaly_data, a0)
    pattern = {}
    # make the u matrix of exponential decays corresponding to the fitted timescales
    u_np = make_amat(aopt.params, len(anomaly_data.time))
    # make it into an 2d xarray object, time by mode
    uxr = xr.DataArray(
        data=u_np,
        dims=["time", "mode"],
        coords={
            "time": (["time"], anomaly_data.time.data),
            "mode": (["mode"], np.arange(n_modes)),
        },
    )
    # store the u matrix in the pattern dictionary
    pattern["u"] = uxr
    # now calculate the penrose inverse of U
    ui = pinv(u_np)
    # now calculate the pattern/v matrix by taking the dot product of the penrose inverse of u with the anomaly data
    b = np.tensordot(ui, anomaly_data.values, axes=1)
    # make an xarray object of the pattern matrix
    bx = xr.DataArray(
        data=b,
        dims=["mode", "lat", "lon"],
        coords={
            "lat": (["lat"], anomaly_data.lat.data),
            "lon": (["lon"], anomaly_data.lon.data),
            "mode": (["mode"], np.arange(n_modes)),
        },
    )
    # store the pattern matrix in the pattern dictionary
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
    nt = len(residual_anom.time)
    pattern = {}
    gmanom = global_mean(residual_anom)
    # TODO: Are all amplitudes = 1 a valid assumption?
    bounds = [(10 ** (mode_num), 10 ** (mode_num + 1)) for mode_num in range(n_modes)]
    opt = minimize(
        rmse_gm,
        [5 * 10 ** (mode_num) for mode_num in range(n_modes)],
        args=(gmanom, fcg_aer),
        bounds=bounds,
        method="Powell",
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
    )[:, : len(fcg_aer)]
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


def recon_separately(pattern_full, pc_matrix):
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
    # Define matrices based on dictionary input:
    mode_timescales = pattern_synth["u"]  # size n_time by n_modes
    pattern_per_mode = pattern_synth["v"]  # size n_pixels by n_modes
    # number of modes
    n_modes = pattern_per_mode.shape[0]
    # reshape v1 into a 2d matrix
    pattern_2d = pattern_per_mode.values.reshape(n_modes, -1)

    recon_per_mode = np.zeros(
        (
            n_modes,
            mode_timescales.shape[0],
            pattern_per_mode.shape[1],
            pattern_per_mode.shape[2],
        )
    )
    for mode in range(n_modes):
        # compute reconstruceted field (unweighted) as dot product
        recon_per_mode[mode, :, :, :] = np.reshape(
            np.outer(mode_timescales[:, mode], pattern_2d[mode, :]),
            [
                mode_timescales.shape[0],
                pattern_per_mode.shape[1],
                pattern_per_mode.shape[2],
            ],
        )
    # convert reconstructed field to xarray and return
    recon_xarray = xr.DataArray(
        recon_per_mode,
        coords=(
            range(n_modes),
            mode_timescales.time,
            pattern_per_mode.lat,
            pattern_per_mode.lon,
        ),
        dims=("mode", "time", "lat", "lon"),
    )
    return recon_xarray


def recon(pattern):
    """
    Reconstruct full dataset given a PCA decompostion represented
    in the dictionary format outputted by eof_calculation_wrapper

    Output in xarray dataarray format (time by lat by lon)

    Parameters
    ----------
    pattern : dict
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
    mode_timescales = pattern["u"]  # size n_time by n_modes
    pattern_per_mode = pattern["v"]  # size n_pixels by n_modes
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
