
# set initial temperature profile to be the surface temperature (isothermal)
rcm.state.Tatm[:] = rcm.state.Ts

# compute diagnostics
_ = rcm.compute_diagnostics()

#  plot initial data
fig, lines = initial_figure(rcm)

# make animation - this animation can take a while
animation.FuncAnimation(fig, animate, 50, fargs=(rcm, lines))