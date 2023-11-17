
# run the model to equilibrium without forcing and begining with T(0) = 288K

# define the number of timesteps (years) to run the model
numtsteps = 40

# set the intial temperature (initial condition)
T_series = [288]

# set the initial time to 0
t_series = [0]

# run the model
for n in range(numtsteps):
    # calculate and append the time since running the model, dependent on dt and the numtsteps
    t_series.append((n + 1) * dt / sec_2_yr)

    # calculate and append the new temperature using our pre-defined function and get energy balance
    T_new, Ftoa = step_forward(T_series[n], alpha=alpha, tau=tau, dt=dt)
    T_series.append(T_new)

print(Ftoa)