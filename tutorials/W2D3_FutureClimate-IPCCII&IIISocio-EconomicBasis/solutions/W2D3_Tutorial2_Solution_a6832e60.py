for prstp in [0.015, 0.005]:
    dice = dicelib.DICE()
    dice.init_parameters(prstp=prstp)
    dice.init_variables()
    controls_start, controls_bounds = dice.get_control_bounds_and_startvalue()
    dice.optimize_controls(controls_start, controls_bounds)
    dice.roll_out(dice.optimal_controls)
    dice.plot_run("discount rate, r=" + str(prstp))