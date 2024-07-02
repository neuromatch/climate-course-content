import numpy as np
from scipy import stats
from scipy.stats import genextreme as gev
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

def estimate_return_level(quantile,loc,scale,shape):
    level = loc + scale / shape * (1 - (-np.log(quantile))**(shape))
    return level

def estimate_return_level_period(period,loc,scale,shape):
    '''
    Compute GEV-based return level for a given return period, and GEV parameters
    '''
    return gev.ppf(1-1/period,shape,loc=loc,scale=scale)

def empirical_return_level(data):
    '''
    Compute empirical return level using the algorithm introduced in Tutorial 2
    '''
    df = pd.DataFrame(index=np.arange(data.size))
    # sort the data
    df['sorted'] = np.sort(data)[::-1]
    # rank via scipy instead to deal with duplicate values
    df['ranks_sp'] = np.sort(stats.rankdata(-data))
    # find exceedence probability
    n = data.size
    df['exceedance'] = df['ranks_sp']/(n+1)
    # find return period
    df['period'] = 1 / df['exceedance']

    df = df[::-1]

    out = xr.DataArray(
        dims=['period'],
        coords={'period':df['period']},
        data=df['sorted'],name='level')
    return out

def fit_return_levels(data,years,N_boot=None,alpha=0.05):
    '''
    Fit GEV to data, compute return levels and confidence intervals
    '''
    empirical = empirical_return_level(data).rename({'period':'period_emp'}).rename('empirical')
    shape, loc, scale = gev.fit(data,0)
    print('Location: %.1e, scale: %.1e, shape: %.1e' % (loc, scale, shape))
    central = estimate_return_level_period(years,loc,scale,shape)

    out = xr.Dataset(
        # dims = ['period'],
        coords = {
            'period': years,
            'period_emp': empirical['period_emp']
            },
        data_vars={
            'empirical':(['period_emp'],empirical.data),
            'GEV':(['period'],central)
            }
    )

    if N_boot:
        levels = []
        shapes, locs, scales = [],[],[]
        for i in range(N_boot):
            datai = np.random.choice(data,size=data.size,replace=True)
            # print(datai.mean())
            shapei,loci,scalei = gev.fit(datai,0)
            shapes.append(shapei)
            locs.append(loci)
            scales.append(scalei)
            leveli = estimate_return_level_period(years,loci,scalei,shapei)
            levels.append(
                leveli
            )

        levels = np.array(levels)
        quant = alpha / 2, 1-alpha/2
        quantiles = np.quantile(levels,quant,axis=0)

        print('')
        print('Ranges with alpha = %.3f :' % alpha)
        print('Location: [%.2f , %.2f]'  % tuple(np.quantile(locs,quant).tolist()))
        print('Scale: [%.2f , %.2f]'  % tuple(np.quantile(scales,quant).tolist()))
        print('Shape: [%.2f , %.2f]'  % tuple(np.quantile(shapes,quant).tolist()))

        quantiles = xr.DataArray(
            dims=['period','quantiles'],
            coords={'period':out.period,'quantiles':np.array(quant)},
            data=quantiles.T
        )
        out['range'] = quantiles
    return out

def plot_return_levels(obj,c='C0',label='',ax=None):
    '''
    Plot fitted data:
        - empirical return level
        - GEV-fitted return level
        - alpha-confidence ranges with bootstrapping (if N_boot is given)
    '''
    if not ax:
        ax = plt.gca()
    obj['GEV'].plot.line('%s-' % c,lw=3,_labels=False,label=label,ax=ax)
    obj['empirical'].plot.line('%so' % c,mec='k',markersize=5,_labels=False,ax=ax)
    if 'range' in obj:
        # obj['range'].plot.line('k--',hue='quantiles',label=obj['quantiles'].values)
        ax.fill_between(obj['period'],*obj['range'].T,alpha=0.3,lw=0,color=c) 
    ax.semilogx()
    # ax.legend()