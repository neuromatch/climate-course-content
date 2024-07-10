fig, ax = plt.subplots()
# multiply Qglobal by 1000 to put in units of grams water vapor per kg of air
_ = ax.plot(Qglobal * 1000.0, Qglobal.lev, label="Specific humidity (g/kg)")
# multiply by 1E6 to get units of ppmv = parts per million by volume
_ = ax.plot(radmodel.absorber_vmr["O3"] * 1e6, radmodel.lev, label="Ozone (ppmv)")

# pressure decreases logarithmically with height in the atmosphere
# invert the axis so the largest value of pressure is lowest
ax.invert_yaxis()
# set y axis to a log scale
_ = ax.set_yscale("log")

ax.set_ylabel("Pressure (hPa)")
ax.set_xlabel("Specific humidity (g/kg)")

# turn on the grid lines
_ = ax.grid()

# turn on legend
_ = ax.legend()