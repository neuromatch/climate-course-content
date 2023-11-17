
# one year expressed in seconds
one_yr = 60.0 * 60.0 * 24.0 * 365.0

# legend labels
labels = ["dt = half-year", "dt = one year", "dt = five years"]

# define the number of timesteps (years) to run the model
numtsteps = np.array([10, 5, 1]) * 3

# for converting number of seconds in a year
sec_2_yr = 3.154e7

fig, ax = plt.subplots()
# loop through each choice of time step
for dd, dt_2 in enumerate([one_yr * 0.5, one_yr, one_yr * 5]):
    # set the intial temperature (initial condition)
    T_series = [288]

    # set the initial time to 0
    t_series = [0]

    # run the model
    for n in range(numtsteps[dd]):
        # calculate and append the time since running the model, dependent on dt and the numtsteps
        t_series.append((n + 1) * dt_2 / sec_2_yr)

        # calculate and append the new temperature using our pre-defined function
        T_series.append(step_forward(T_series[n], alpha=alpha, tau=tau, dt=dt_2))

    ax.plot(t_series, T_series, label=labels[dd])

ax.set_xlabel("Years")
ax.set_ylabel("Global mean temperature (K)")
ax.legend()