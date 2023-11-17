for a3 in [2, 3]:
    dice = dicelib.DICE()
    dice.init_parameters(a3=a3)
    dice.init_variables()
    controls_start, controls_bounds = dice.get_control_bounds_and_startvalue()
    dice.optimize_controls(controls_start, controls_bounds)
    dice.roll_out(dice.optimal_controls)
    dice.plot_run("damage function exponent, a3=" + str(a3))