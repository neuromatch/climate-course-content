fig, ax = plt.subplots(1, 2)
max_year = 2500
TT = dice_std.TT
NT = dice_std.NT
upp, low = zip(*controls_bounds_std[:NT])
ax[0].plot(TT, dice_std.optimal_controls[:NT], label="optimal")
ax[0].plot(TT, upp, "k--", label="bounds")
ax[0].plot(TT, low, "k--")
ax[0].set_ylabel("mitigation rate")
# Set limits
_ = ax[0].set_xlim(2000, max_year)
ax[0].legend(frameon=False)
upp, low = zip(*controls_bounds_std[NT:])
ax[1].plot(TT, dice_std.optimal_controls[NT:])
ax[1].plot(TT, upp, "k--")
ax[1].plot(TT, low, "k--")
ax[1].set_ylabel("savings rate")
ax[1].set_xlabel("year")
# Set limits
_ = ax[1].set_xlim(2000, max_year)

fig.tight_layout()