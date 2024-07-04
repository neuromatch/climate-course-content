
# put two variables of interest in a list
vars = ['Emissions|Kyoto Gases', 'Land Cover|Forest']
# create new names for structured data series and plot labels
val_name = ['Emissions\n(Mt CO$_2$/yr)','Land covered by\nforest (million ha)']
# choose scenarios of interest and a color for plotting
scenarios = ['SSP1-26', 'SSP5-Baseline']
colors = ['darkblue','darkorange']


# init figure and axis
fig, axs = plt.subplots(2,1)
# loop over all variables and new names
for var, val, ax in zip(vars,val_name, axs.flatten()):

    # loop over scenarios and their color
    for sc, col in zip(scenarios, colors):
        # retrieve SSP for the respective variable from rich dataframe
        ds_unstrct = get_SSPs_for_variable(df,sc,var)
        # restructure dataframe for plotting
        ds_strct = pd.melt(ds_unstrct, id_vars=["MODEL"], value_vars=['2010','2020','2030','2040','2050','2060','2070','2080','2090','2100'], var_name="YEAR", value_name =val)
        #print(ds_strct)
        # plot variable vs. time, add label incl. scenario and model
        ax.plot(ds_strct['YEAR'],ds_strct[val],label=f'{sc},\n{ds_strct.MODEL[0]}', color=col)
        # altern. plotting procedure w/o the color distinction
        #sns.lineplot(ds_strct, x='YEAR', y=val, hue='MODEL', ax=ax, palette='flare')

    # aesthetics
    ax.set_ylabel(fr'{val}')
    ax.set_xlabel('Time (years)')
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.setp(ax.get_xticklabels()[::2], visible=False)
    ax.grid(True)
    axs[0].legend()