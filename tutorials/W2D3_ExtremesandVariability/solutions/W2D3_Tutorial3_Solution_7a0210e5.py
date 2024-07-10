"""
The three parameters of the Generalized Extreme Value (GEV) distribution—location, scale, and shape—each impact the shape of the distribution in unique ways:

Location parameter: This parameter shifts the distribution along the x-axis. In the context of extreme events, changing the location parameter shifts the entire distribution of extreme values, affecting the threshold at which extreme events are defined. For example, increasing the location parameter would imply that previoously already extreme events are occurring at higher values.

Scale parameter: The scale parameter influences the spread or width of the distribution. A higher scale parameter will widen the distribution, implying greater variability in the data and thicker tails, while a lower scale parameter will narrow the distribution, implying less probability of extreme events.

Shape parameter: This parameter is unique to the GEV distribution and impacts the tails of the distribution. It can make the tails of the distribution thinner or thicker, which in turn affects the skewness and kurtosis of the distribution. In the context of extreme events, a positive shape parameter results in a heavy-tailed distribution, implying a higher probability of very extreme events (far from the median), while a negative shape parameter results in a light-tailed distribution, implying a lower probability of such events.
"""