
# define your functions and constants

# define albedo
alpha = 0.2941  # unitless number between 0 and 1 (calculated previously from observations in tutorial 2)

# define transmissivity (calculated previously from observations in tutorial 1)
tau = 0.6127    # unitless number between 0 and 1

# effective radiative forcing for a doubling of CO2
F = 3.93  # W/m^2

# define the time interval, one year expressed in seconds
dt = 60.0 * 60.0 * 24.0 * 365.0

# for converting number of seconds in a year
sec_2_yr = 3.154e7


# define a function for absorbed shortwave radiation (ASR)
def ASR(alpha, Q):
    return (1 - alpha) * Q


# define a function for outgoing longwave radiation (OLR)
def OLR(tau, T):
    # define the Stefan-Boltzmann Constant, noting we are using 'e' for scientific notation
    sigma = 5.67e-8  # W m^-2 K^-4

    return tau * sigma * T**4


# create a function to find the new temperature based on the previous using Euler's method.
def step_forward(T, alpha, tau, dt):
    # define the observed insolation based on observations from the IPCC AR6 Figure 7.2
    Q = 340  # W m^-2

    Ftoa = ASR(alpha, Q) - OLR(tau, T)

    T_new = T + dt / C * Ftoa

    return T_new, Ftoa


# create a function to find the new temperature based on the previous using Euler's method.
def step_forward_forced(T, alpha, tau, dt):
    # define the observed insolation based on observations from the IPCC AR6 Figure 7.2
    Q = 340  # W m^-2

    # define the Stefan-Boltzmann Constant, noting we are using 'e' for scientific notation
    sigma = 5.67e-8  # W m^-2 K^-4

    Ftoa = ASR(alpha, Q) - OLR(tau, T) + F

    T_new = T + dt / C * Ftoa

    return T_new, Ftoa