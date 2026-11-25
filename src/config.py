import numpy as np


MASS_DRONE = 16.5  # [kg]
MASS_PAYLOAD = 5.65  # [kg]
MASS_TOTAL = MASS_DRONE + MASS_PAYLOAD
GRAVITY = 9.81  # [m/s^2]
TETHER_LEN = 9.6  # [m]
J = np.diag([0.08, 0.08, 0.129])  # [kg m^2]
