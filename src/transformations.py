"""
Transformations library

I   inertial (ENU: east, north, up)
B   body (ENU ordered: starboard, nose, up)
C   camera
P   payload

Functions are named T_AB: rotation to frame A from frame B, so
T_AB @ v_B == v_A.
"""
import numpy as np


def T_IB(phi, theta, psi) -> np.ndarray:
    """
    Rotation to the inertial frame from the body frame
    """
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)
    sy, cy = np.sin(psi), np.cos(psi)

    T = np.array([[ct*cy, sp*st*cy - cp*sy, cp*st*cy + sp*sy],
                 [ct*sy, sp*st*sy + cp*cy, cp*st*sy - sp*cy],
                 [-st, sp*ct, cp*ct]])

    return T
