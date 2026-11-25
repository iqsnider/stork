from . import config
import numpy as np


class ArduPilotFlightController:
    """
    "ME236" classic cascaded control system for the inner-loop

    Accepts acceleration in the model's +Z-up frame and a yaw setpoint.
    The small-angle roll/pitch mapping assumes yaw is near zero.
    """

    def __init__(self,
                 tau_phi=0.3,
                 tau_theta=0.3,
                 tau_psi=0.5,
                 tau_p=0.05,
                 tau_q=0.05,
                 tau_r=0.08) -> None:

        self.tau_theta = tau_theta
        self.tau_phi = tau_phi
        self.tau_psi = tau_psi
        self.tau_p = tau_p
        self.tau_q = tau_q
        self.tau_r = tau_r

    def compute_u(self, x, a_des, yaw_s) -> list[float]:
        """
        Computes [C_Sigma, n1, n2, n3] from the acceleration and yaw setpoint
        """
        g = config.GRAVITY
        m = config.MASS_TOTAL
        Jx = config.J[0, 0]
        Jz = config.J[2, 2]

        phi = x[6]
        theta = x[7]
        psi = x[8]
        p = x[9]
        q = x[10]
        r = x[11]

        a1, a2, a3 = a_des

        # thrust calculation
        C_Sigma = m*np.sqrt(a1**2 + a2**2 + (a3 + g)**2)

        # acceleration transform
        theta_s = a1*m/C_Sigma
        phi_s = -a2*m/C_Sigma
        self.att_s = np.array([phi_s, theta_s])  # kept for a no lag sim

        # attitude control
        p_s = (1/self.tau_phi)*(phi_s - phi)
        q_s = (1/self.tau_theta)*(theta_s - theta)
        r_s = (1/self.tau_psi)*(yaw_s - psi)

        # angular velocity control
        n1 = (Jx/self.tau_p)*(p_s - p) + (Jx - Jz)*q*r
        n2 = (Jx/self.tau_q)*(q_s - q) + (Jz - Jx)*p*r
        n3 = (Jz/self.tau_r)*(r_s - r)

        return [C_Sigma, n1, n2, n3]
