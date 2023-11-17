
# define the Stefan-Boltzmann Constant, noting we are using 'e' for scientific notation
sigma = 5.67e-8  # W m^-2 K^-4

# define the global mean surface temperature based on observations
T = 288  # K

# plug into equation
OLR = sigma * (T**4)

# display answer
print("OLR: ", OLR, "W m^2")