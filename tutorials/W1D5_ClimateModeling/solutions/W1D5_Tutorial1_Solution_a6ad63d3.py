
# define the Stefan-Boltzmann Constant, noting we are using 'e' for scientific notation
sigma = 5.67e-8  # W m^-2 K^-4

# define the emission temperature based on observtions of global mean surface temperature
T = 288  # K

# define values of tau
tau = [0.2, 0.6127, 0.8]

# get values of OLR from tau using list comprehension
OLR = [t * sigma * (T**4) for t in tau]

# convert tau to list of strings using list comprehension so we can create a categorical plot
tau = [str(t) for t in tau]

fig, ax = plt.subplots()
_ = ax.bar(tau, OLR, color="#67ada9")
ax.set_xlabel("Transmissivity")
ax.set_ylabel("Outgoing Longwave Radiation (W m$^{-2}$)")