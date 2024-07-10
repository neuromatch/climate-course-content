
"""
1. It will affect the solution our model provides. Euler's method is a linear approximation of the 'truth', and thus if the true answer (say analytically found) is very non-linear than our solution will be very sensitive to the time step used. Generally a smaller time step will provide a more accurate solution (i.e. closer to the 'truth'). With more complex models, there is also a tradeoff with computing resources where adding shorter timesteps can significantly increase the computation time of the desired length of simulation.
"""