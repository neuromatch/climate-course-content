import numpy as np
import SDFC as sd
import xarray as xr
import matplotlib.pyplot as plt
import texttable as tt

import warnings
warnings.filterwarnings('ignore')

def return_period_obs(da,periods_per_year,threshold=None):
    '''
    Compute empirical return levels for 1D time series
    Returns L = L(T) with L return levels in units of the variable provided and and T return periods in years
    - da: 1D array or DataArray, will be converted to numpy array for processing
    - periods_per_year: number of timesteps per year
    - threshold: optional: only returns only events above a given return level
    '''
    if not isinstance(da,np.ndarray):
        da_values_sorted = np.sort(da.values.flatten())
    else:
        da_values_sorted = np.sort(da.flatten())
    # total number of points
    N = da_values_sorted.size
    # observations exceeding threshold
    if threshold is not None:
        da_values_sorted = da_values_sorted[np.where(da_values_sorted>threshold)]
    # fraction of observations exceeding threshold
    k = da_values_sorted.size
    zeta_u = k/N

    # Exceedance probabilities for the observations (>threshold)
    prob_exceed = (k - np.arange(1,k+1) + 1) / (k + 1)
    # Calculate return periods in years from exceedance probabilities and timestepping
    periods = 1 / (prob_exceed * periods_per_year * zeta_u )
    out = xr.DataArray(dims=['return period'],coords={'return period':periods},data=da_values_sorted,name='return level')
    out['return period'].attrs['units'] = 'year'
    return out

def print_law( law ):
    '''
    Print fitted SDFC model
    '''
    tab = tt.Texttable( max_width = 0 )
    
    ## Header
    header = [ str(type(law)).split(".")[-1][:-2] + " ({})".format(law.method) , "Type" , "coef" ]
    if hasattr(law.info_,'coefs_ci_bs_'):
        header += [ "Quantile {}".format(law.info_.alpha_ci_/2) , "Quantile {}".format( 1 - law.info_.alpha_ci_ / 2 ) ]
    tab.header( header )
    
    ## Loop on params

    covariates = [c is not None for c in law._rhs.c_global]
    if covariates == [0,0,0]:
        idx = [0,1,2]
    elif covariates == [1,0,0]:
        idx = [[0,1],2,3]
    elif covariates == [0,1,0]:
        idx = [0,[1,2],3]
    elif covariates == [0,0,1]:
        idx = [0,1,[2,3]]
    elif covariates == [1,1,0]:
        idx = [[0,1],[2,3],4]
    elif covariates == [1,0,1]:
        idx = [[0,1],2,[3,4]]
    elif covariates == [0,1,1]:
        idx = [0,[1,2],[3,4]]
    elif covariates == [1,1,1]:
        idx = [[0,1],[2,3],[4,5]]

    for p in law._lhs.names:
        i = law._lhs.names.index(p)
        if covariates[i]:
            label = 'Covariate'
            coef = law.coef_.round(3)[idx[i]].tolist()
        else:
            label = 'Stationary'
            coef = law.coef_.round(3)[idx[i]]
        
        row = [ p , label , coef ]
        if hasattr(law.info_,'coefs_ci_bs_'):
            if not law._lhs.is_fixed(p):
                row += [ law.info_.coefs_ci_bs_[j,idx[i]].squeeze().round(3).tolist() for j in range(2) ]
            else:
                row += [ str(None) , str(None) ]
        tab.add_row( row )
    print(tab.draw() + "\n")

def fit_return_levels_sdfc(da,times,periods_per_year,kind,N_boot=None,full=False,model=False,method='mle',**kwargs):
    '''
    Fit data to GPD or GEV and return results
    Inputs:
        - da: 1D DataArray of numpy array, timeseries
        - threshold: threshold for GPD
        - times: return times in years for which to compute return levels, 1D Array

    2013/10/13: drop NaNs from array before computing
    '''
    # fix covariates
    for key,value in kwargs.items():
        if 'c_' in key:
            kwargs[key] = value[:,np.newaxis]

    try:
        units = da.attrs['units']
    except (AttributeError,KeyError):
        units = ''
    # gpd = fit_gpd(da,threshold,N_boot=N_boot)
    if not isinstance(da,np.ndarray):
        assert len(da.dims) == 1
        dim = da.dims[0]
        # Y = da.values
        Y = da.dropna(dim).values
    else:
        assert len(da.shape) == 1
        Y = da
        Y = da[~np.isnan(da)]

    # Fitting using SDFC
    if kind.upper() == 'GPD':
        threshold = kwargs['f_loc']
        law_gpd = sd.GPD(method = method.lower())
        if N_boot:
            law_gpd.fit_bootstrap(Y,n_bootstrap=N_boot,alpha=0.05,**kwargs)
        else:
            law_gpd.fit_bootstrap(Y,**kwargs)
        # law_gpd.fit(Y, **kwargs)
        # law_gpd.fit(Y, f_loc = threshold,**kwargs)
        zeta_u = Y[Y>threshold].size / Y.size # fraction of points exceeding threshold
        # According to Coles 2001, Eq. 4.13 ff - sign for xi NOT reversed, SDFC has same convention as Coles
        # if law_gpd.n_bootstrap == 0:
        if not N_boot:
            return_levels = threshold + law_gpd.coef_[0] /  law_gpd.coef_[1]  * (( times[:,None] * periods_per_year * zeta_u )**law_gpd.coef_[1] - 1)

            out = xr.DataArray(dims=['return period'],coords={'return period':times},data=np.squeeze(return_levels),name='return level')
        else:
            if not law_gpd._lhs.is_fixed('scale') and not law_gpd._lhs.is_fixed('shape'): # fit both scale and shape
                return_levels = threshold + law_gpd.info_.coefs_bs_[:,0] /  law_gpd.info_.coefs_bs_[:,1]  * (( times[:,None] * periods_per_year * zeta_u )**law_gpd.info_.coefs_bs_[:,1] - 1)
            elif not law_gpd._lhs.is_fixed('scale') and law_gpd._lhs.is_fixed('shape'): # fit only scale, shape is fixed
                return_levels = threshold + law_gpd.info_.coefs_bs_[:,0] /  kwargs['f_shape']  * (( times[:,None] * periods_per_year * zeta_u )**kwargs['f_shape'] - 1)
            elif law_gpd._lhs.is_fixed('scale') and not law_gpd._lhs.is_fixed('shape'): # scale is fixed, fit only shape
                return_levels = threshold + kwargs['f_scale'] /  law_gpd.info_.coefs_bs_[:,0]  * (( times[:,None] * periods_per_year * zeta_u )**law_gpd.info_.coefs_bs_[:,0] - 1)

            N = np.arange(law_gpd.info_.n_bootstrap)
            out = xr.DataArray(dims=['return period','N'],coords={'return period':times,'N':N},data=return_levels,name='return level')
            out['N'].attrs['long_name'] = 'Number of bootstrapping samples'

        out['return period'].attrs['units'] = 'year'
        out.attrs['units'] = units
        out.attrs['zeta_u'] = zeta_u
        out.attrs['kind'] = kind
        out.attrs['method'] = method

        if full is True:
            out = out.to_dataset()
            if N_boot is None:
                out['mu'] = threshold
                out['sigma'] = law_gpd.coef_[0]
                out['xi'] = law_gpd.coef_[1]
            else:
                out['mu'] = xr.DataArray(dims=['N'],coords={'N':out['N']},data=threshold * np.ones(N_boot))

                if not law_gpd._lhs.is_fixed('scale'):
                    out['sigma'] = xr.DataArray(dims=['N'],coords={'N':out['N']},data=law_gpd.info_.coefs_bs_[:,0])
                else:
                    out['sigma'] = xr.DataArray(dims=['N'],coords={'N':out['N']},data=kwargs['f_scale'] * np.ones(N_boot))

                if law_gpd._lhs.is_fixed('shape'):
                    out['xi'] = xr.DataArray(dims=['N'],coords={'N':out['N']},data=kwargs['f_shape'] * np.ones(N_boot))
                elif not law_gpd._lhs.is_fixed('shape') and law_gpd._lhs.is_fixed('scale'):
                    out['xi'] = xr.DataArray(dims=['N'],coords={'N':out['N']},data=law_gpd.info_.coefs_bs_[:,0])
                else:
                    out['xi'] = xr.DataArray(dims=['N'],coords={'N':out['N']},data=law_gpd.info_.coefs_bs_[:,1])
            out['return_level_obs'] = return_period_obs(da,periods_per_year,threshold=threshold).rename({'return period':'return_period_obs'})
        if model is True:
            return out, law_gpd
        else:
            return out
    elif kind.upper() == 'GEV':
        law_gev = sd.GEV(method = method.lower())
        if N_boot:
            law_gev.fit_bootstrap(Y,n_bootstrap=N_boot,alpha=0.05,**kwargs)
        else:
            law_gev.fit(Y,**kwargs)

        # zeta_u = Y[Y>threshold].size / Y.size # fraction of points exceeding threshold
        # According to Coles 2001, Eq. 4.13 ff

        yp = -np.log(1 - 1/times)

        if not N_boot:
            loc, scale, shape = law_gev.coef_
            mu, sigma, xi = loc, scale, shape
            # Coles, 3.10 (page 56, section 3.3.3)
            return_levels = loc - scale / shape  * ( 1 - yp[:,None]**(-shape))
            out = xr.DataArray(dims=['return period'],coords={'return period':times},data=np.squeeze(return_levels),name='return level')
        else:
            N = np.arange(law_gev.info_.n_bootstrap) # bootstrap coordinate
            params_fixed =  [law_gev._lhs.is_fixed('loc'), law_gev._lhs.is_fixed('scale'), law_gev._lhs.is_fixed('shape')]
            if params_fixed == [True, True, True]: # fixed all 
                print('Error - cannot fix all parameters')
                return None
            elif params_fixed == [True, True, False]: # fixed location, scale
                loc, scale, shape = kwargs['f_loc'], kwargs['f_scale'], law_gev.info_.coefs_bs_[:,0]
                mu, sigma, xi = loc, scale, xr.DataArray(dims=['N'],coords={'N':N},data=shape,name='xi')
            elif params_fixed == [True, False, True]: # fixed location, shape
                loc, scale, shape = kwargs['f_loc'], law_gev.info_.coefs_bs_[:,0], kwargs['f_shape']
                mu, sigma, xi = loc, xr.DataArray(dims=['N'],coords={'N':N},data=scale,name='sigma'), shape
            elif params_fixed == [False, True, True]: # fixed scale, shape
                loc, scale, shape = law_gev.info_.coefs_bs_[:,0], kwargs['f_scale'], kwargs['f_shape']
                mu, sigma, xi = xr.DataArray(dims=['N'],coords={'N':N},data=loc,name='mu'), scale, shape
            elif params_fixed == [False, True, False]: # fixed scale
                loc, scale, shape = law_gev.info_.coefs_bs_[:,0], kwargs['f_scale'], law_gev.info_.coefs_bs_[:,1]
                mu, sigma, xi = xr.DataArray(dims=['N'],coords={'N':N},data=loc,name='mu'), scale, xr.DataArray(dims=['N'],coords={'N':N},data=shape,name='xi')
            elif params_fixed == [False, False, True]: # fixed shape
                loc, scale, shape = law_gev.info_.coefs_bs_[:,0], law_gev.info_.coefs_bs_[:,1], kwargs['f_shape']
                mu, sigma, xi = xr.DataArray(dims=['N'],coords={'N':N},data=loc,name='mu'), xr.DataArray(dims=['N'],coords={'N':N},data=scale,name='sigma'), shape
            elif params_fixed == [False, False, False]: # fixed none of loc, scale, shape
                loc, scale, shape = law_gev.info_.coefs_bs_.T
                mu, sigma, xi = xr.DataArray(dims=['N'],coords={'N':N},data=loc,name='mu'), xr.DataArray(dims=['N'],coords={'N':N},data=scale,name='sigma'), xr.DataArray(dims=['N'],coords={'N':N},data=shape,name='xi')

            return_levels = loc - scale / shape  * ( 1 - yp[:,None]**(-shape))
            out = xr.DataArray(dims=['return period','N'],coords={'return period':times,'N':N},data=return_levels,name='return level')
            out['N'].attrs['long_name'] = 'Number of bootstrapping samples'

        out['return period'].attrs['units'] = 'year'
        out.attrs['units'] = units
        # out.attrs['zeta_u'] = zeta_u
        out.attrs['kind'] = kind
        out.attrs['method'] = method

        if full is True:
            out = out.to_dataset()
            out['mu'] = mu
            out['sigma'] = sigma
            out['xi'] = xi
            # out['return_level_obs'] = return_period_obs(da,periods_per_year,threshold=threshold).rename({'return period':'return_period_obs'})
            out['return_level_obs'] = return_period_obs(da,periods_per_year).rename({'return period':'return_period_obs'})
        if model is True:
            return out, law_gev
        else:
            return out
    else:
        raise ValueError('kind %s is not defined' % kind)
    
def fit_return_levels_sdfc_2d(da,times,periods_per_year,kind,N_boot,percentile=None,full=False,method='mle',**kwargs):
    '''
    Iterate over latitude, longitude and fit indendently at each location, same threshold
    - full: also get obs and parameters at each location
    - need ONLY one of threshold, percentile
        if threshold: fixed threshold for each point
        if percentile: fixed percentile for each point
    - method: 
        use SDFC MLE ('MLE') or L-Moments ('LM')
    - fixed parameters (f_loc, f_scale, f_shape) set in kwarsgs
        those are either:
            -single float value - then the parameter is set for the entire 2d region
            -dataarray with same grid as da - then the parameter is set per gridpoint
    '''
    if kind.upper() == 'GPD':
        if not 'f_loc' in kwargs.keys() and percentile is not None:
            print('Fixed percentile')
        elif 'f_loc' in kwargs.keys() and percentile is None:
            print('Fixed threshold')
        else:
            print('GPD: ERROR: Need to set ONLY ONE of threshold, percentile')
            return 

    func = fit_return_levels_sdfc
    tmps = []
    i = 1
    for lati in da['latitude'].values:
        print('Latitude: %i / %i : %.1f' % (i,da['latitude'].size,lati))
        i += 1
        tmpsi = []
        for loni in da['longitude'].values:
            dai = da.sel(latitude=lati,longitude=loni)
            kwargs2 = {}
            for key in kwargs.keys():
                if isinstance(kwargs[key],xr.DataArray):
                    kwargs2[key] = float(kwargs[key].sel(latitude=lati,longitude=loni))
                else:
                    kwargs2[key] = kwargs[key]

            if kind.upper() == 'GPD' and percentile is not None:
                kwargs2['f_loc'] = np.quantile(dai.values,percentile)
            # tmp = ex.fit_return_levels(dai,threshold=threshold,times=times,periods_per_year=periods_per_year,N_boot=N_boot,full=full)
            tmp = func(dai,times=times,periods_per_year=periods_per_year,kind=kind,N_boot=N_boot,full=full,method=method,**kwargs2)
            # tmp['longitude'] = loni
            # # tmp['latitude'] = lati
            # tmpsi.append(tmp)
            # print(loni)
            # # return tmp
            try:
                # tmp['N']
                tmp['longitude'] = loni
                # tmp['latitude'] = lati
                tmpsi.append(tmp)
            except:
                print('Error, no N')

        tmpsi = xr.concat(tmpsi,'longitude')
        tmpsi['latitude'] = lati
        tmps.append(tmpsi)
    tmps = xr.concat(tmps,'latitude')
    return tmps
    
def plot_levels_from_obj(da,ax=None,alpha=None,lw=3,c='C0',obs=True,marker='o',markersize=5,mec='k',**kwargs):
    '''
    Plot results from 1D object returned from fit_return_levels
    '''
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    if isinstance(da,xr.DataArray): # not full output, only 'return level'
        if not 'N' in da.dims:
            da.plot.line('%s-' % c,lw=lw,ax=ax,**kwargs)
        else:
            da.median('N').plot.line('%s-' % c,lw=lw,ax=ax,**kwargs)
            if alpha is not None:
                lower, upper = [(1 - alpha)/2,alpha + (1-alpha)/2]
                ax.fill_between(da['return period'],*da.quantile([lower,upper],'N'),alpha=0.3,color=c)
    elif isinstance(da,xr.Dataset): # full output, including observed values and GPD parameters
        if not 'N' in da.dims:
            da['return level'].plot.line('%s-' % c,lw=lw,ax=ax,**kwargs)
        else:
            da['return level'].median('N').plot.line('%s-' % c,lw=lw,ax=ax,**kwargs)
            if alpha is not None:
                lower, upper = [(1 - alpha)/2,alpha + (1-alpha)/2]
                ax.fill_between(da['return period'],*da['return level'].quantile([lower,upper],'N'),alpha=0.3,color=c,lw=0)
        if obs is True:
            da['return_level_obs'].plot.line(marker,markersize=markersize,color=c,mec=mec,ax=ax,_labels=False) # ,zorder=-1

    ax.semilogx()

    if not fig is None:
        return fig,ax