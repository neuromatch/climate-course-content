
# define the observed insolation based on observations from the IPCC AR6 Figure 7.2
Q = 340  # W m^-2

# define the observed reflected radiation based on observations from the IPCC AR6 Figure 7.2
F_ref = 100  # W m^-2

# define albedo
alpha = F_ref / Q  # unitless number between 0 and 1

# define the Stefan-Boltzmann Constant, noting we are using 'e' for scientific notation
sigma = 5.67e-8  # W m^-2 K^-4


# define a function that returns the equilibrium temperature and takes argument tau
def get_eqT(tau):
    return (((1 - alpha) * Q) / (tau * sigma)) ** (1 / 4)


# define tau as an array extending from 0 to 1 with spacing interval 0.01
tau = np.arange(0, 1.01, 0.01)

# use list comprehension to obtain the equilibrium temperature as a function of tau
eqT = [get_eqT(t) for t in tau]

fig, ax = plt.subplots()
# Plot tau vs. eqT
_ = ax.plot(tau, eqT, lw=3)
ax.set_xlabel("Transmissivity")
ax.set_ylabel("Equilibrium Temperature")