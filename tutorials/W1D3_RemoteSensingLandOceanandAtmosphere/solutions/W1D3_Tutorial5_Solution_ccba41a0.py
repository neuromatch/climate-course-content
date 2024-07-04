
"""

1. The 24-month rolling average is even smoother and some of the 'bumps' rather than 'spikes' are removed.
2. The window size you use would largely depend on the time scales of interest. For example, to calculate hydrological drought mostly caused by rainfall deficit you could use different windows ranging from 3 months to 60 months (https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-index-spi). Averaging over a longer time period will remove smaller-scale variability and allow you to focus on longer-term patterns that might be related to annual or decadal patterns (e.g., the Pacific Decadal Oscillation).

"""