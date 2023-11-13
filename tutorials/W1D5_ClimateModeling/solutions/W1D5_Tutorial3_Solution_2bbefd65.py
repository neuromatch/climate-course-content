
# heat capacity of the atmosphere
c_atm = 1004  # specific heat of the atmosphere at constant pressure in J/kg/K
W_atm = 100000  # weight (pressure) of atmospheric column in Pa
g = 9.81  # height of atmosphere in m (representative of )
C_atm = c_atm * (W_atm / g)  # heat capacity of the atmosphere

# find the depth of the ocean for equivalent atmospheric heat capacity
c_oc = 3850  # specific heat of seawater in J/kg/K
rho_oc = 1025  # average density of seawater in kg/m3

d_oc = C_atm / (c_oc * rho_oc)  # heat capacity of the ocean
d_oc