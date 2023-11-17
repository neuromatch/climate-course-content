
# run the model to equilibrium without forcing and begining with T(0) = 288K

# define the number of timesteps (years) to run the model
numtsteps = 40

# set initial condition (temperature) to the equilibrium value from the last run without forcing
T_series_forced = [T_series[-1]]

# set the initial time to 0
t_series_forced = [0]

# run the model
for n in range(numtsteps):
    # calculate and append the time since running the model, dependent on dt and the numtsteps
    t_series_forced.append((n + 1) * dt / sec_2_yr)

    # calculate and append the new temperature using our pre-defined function and get energy balance
    T_new_forced, Ftoa_forced = step_forward_forced(
        T_series_forced[n], alpha=alpha, tau=tau, dt=dt
    )
    T_series_forced.append(T_new_forced)

print(Ftoa_forced)