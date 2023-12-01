import numpy as np
import numpy.linalg as LA
from Utils import *
import matplotlib.pyplot as plt

import math
from math import atan2, asin, exp, sin, cos, pi

from scipy.integrate import solve_ivp
from functools import partial

sign = lambda x: math.copysign(1, x)

class QuadVTOL:
    """
    Simulate dynamics for a tiltrotor VTOL
    
    State : [pos, vel, rot, pqr]
    Input : [elevons, rpm, rotor_angle]

    """
    def __init__(self):
        self.n = 21
        # Physical parameters
        self.g = 9.81 # [m/s^2]
        self.e3 = np.array([0.,0.,1.]).reshape(3,1)
        self.mass = 0.771 # [kg]
        self.J = np.array([
            [0.0165,    0.0,    4.8e-5], 
            [0.0,       0.0128, 0.0],
            [4.8e-5,    0.0,    0.0282]
        ]) # [kg m^2]
        self.J_inv = LA.inv(self.J)
        self.S = 0.2589 # [m^2]
        self.span = 1.4224 # [m]
        self.chord = 0.3305 # [m]
        self.rho = 1.2682 # [kg/m^3]
        self.es = 0.9 # []
        self.AR = self.span**2 / self.chord # TODO Get actual values
        self.rotor_diameter = 0.18 # [0.18 m]

        # Aerodynamic parameters
        self.set_aero_params()

        # Rotor parameters
        self.set_rotor_params()

    def set_aero_params(self):
        self.CL0 = 0.005
        self.CD0 = 0.0022
        self.CM0 = 0.0
        self.CLa = 2.819
        self.CDa = 0.003
        self.CMa = -0.185
        self.CLq = 3.242
        self.CDq = 0.0
        self.CMq = -1.093
        self.CDp = 0.0
        self.CLdE = 0.2
        self.CDdE = 0.005
        self.CMdE = -0.387

        self.alpha0 = 0.47
        self.sigma_d = 50

        self.CY0 = 0.0
        self.CL0 = 0.0
        self.CN0 = 0.0
        self.CYb = -0.318
        self.CLb = -0.032
        self.CNb = 0.112
        self.CYp = 0.078
        self.CLp = -0.207
        self.CNp = -0.053
        self.CYr = 0.288
        self.CLr = 0.036
        self.CNr = -0.104
        self.CYdE = 0.000536
        self.CLdE = 0.018
        self.CNdE = -0.00328

    def set_rotor_params(self):
        self.CT0 = 0.1167
        self.CT1 = 0.0144
        self.CT2 = -0.1480
        self.CQ0 = 0.0088
        self.CQ1 = 0.0129
        self.CQ2 = -0.0216
        self.rotor_pos = np.array([
            [0.1, 0.1, -0.22],
            [0.18, 0.18, 0],
            [0, 0, 0]
        ]) # TODO Get actual values
        self.k_tilt = 100 # TODO Get actual values

    def compute_sigma(self, alpha):
        d = self.sigma_d
        return (
            (1 + exp(-d*(alpha - self.alpha0)) + exp(d*(alpha - self.alpha0))) /
            (1 + exp(-d*(alpha - self.alpha0))) / (1 + exp(d*(alpha - self.alpha0)))
        )

    def CL(self, alpha):
        sigma = self.compute_sigma(alpha)
        return (1 - sigma) * (self.CL0 + self.CLa * alpha) + sigma*(2*sign(alpha) * sin(alpha)**2 * cos(alpha))

    def CD(self, alpha):
        return self.CDp + (self.CL0 + self.CLa * alpha)**2 / pi/self.es/self.AR

    def aero_forces_and_moments(self, v_a, pqr, dE):
        b = self.span
        c = self.chord
        S = self.S

        # Compute aerodynamic state
        alpha = atan2(v_a[2].item(), v_a[0].item())
        if LA.norm(v_a) > 1e-5:
            beta = asin(v_a[1].item() / LA.norm(v_a))
        else:
            beta = 0.0
        Ra = np.array([
            [np.cos(alpha), 0,  -np.sin(alpha) ],
            [0,             1,  0              ],
            [np.sin(alpha), 0,  np.cos(alpha)  ]
        ])

        # Forces
        F0 = 0.5 * self.rho * np.abs(v_a)**2 * S * Ra @ np.array([
            -c * (self.CD(alpha)),
             b * (self.CY0 + self.CYb * beta),
            -c * (self.CL(alpha))
        ]).reshape(3,1)

        F_omega = 0.4 * self.rho * np.abs(v_a) * S * Ra @ np.array([
            [0,                 c**2 * self.CDq,    0               ],
            [b**2 * self.CYp,   0,                  b**2*self.CYr   ],
            [0,                 c**2 * self.CLq,    0               ]
        ])

        F_dE = 0.5 * self.rho * np.abs(v_a)**2 * S * Ra @ np.array([
            [ c*self.CDdE, -c*self.CDdE ],
            [ b*self.CYdE,  b*self.CYdE ],
            [ c*self.CLdE, -c*self.CLdE ]
        ])
        
        # Moments
        M0 = 0.5 * self.rho * np.abs(v_a)**2 * S * Ra @ np.array([
            b * (self.CL0 + self.CLb * beta),
            c * (self.CM0 + self.CMa * alpha),
            b * (self.CN0 + self.CNb * beta)
        ]).reshape(3,1)

        M_omega = 0.4 * self.rho * np.abs(v_a) * S * Ra @ np.array([
            [b**2 * self.CLp,   0,                  b**2 * self.CLr ],
            [0,                 c**2 * self.CMq,    0               ],
            [b**2 * self.CNp,   0,                  b**2 * self.CNr ]
        ])

        M_dE = 0.5 * self.rho * np.abs(v_a)**2 * S * Ra @ np.array([
            [ self.CLdE, self.CLdE ],
            [-self.CMdE, self.CMdE ],
            [ self.CNdE, self.CNdE ]
        ])

        F_aero = F0 + F_omega @ pqr + F_dE @ dE
        M_aero = M0 + M_omega @ pqr + M_dE @ dE

        return F_aero, M_aero

    def rotor_forces_and_moments(self, v_a, tet_r, rpm):
        # Rot matrix from rotor to body frame
        tet_r = tet_r.ravel()
        rotor_R_body = np.array([
            [np.cos(tet_r[0]),  np.cos(tet_r[1]),   np.cos(tet_r[2])],
            [0,                 0,                  0],
            [-np.sin(tet_r[0]), -np.sin(tet_r[1]),  -np.sin(tet_r[2])]
        ])

        D = self.rotor_diameter
        # Airspeed through rotor i
        # Tranpose to get (3,1) to match rpm dim
        Va = (v_a.T @ rotor_R_body).T

        # rpm (3,1) --> T (3,1) / Q (3,1)
        T = ((self.rho * D**4 * self.CT0) /(4*np.pi**2) * rpm**2 
            + (self.rho * D**3 * self.CT1)/(2*np.pi) * Va * rpm**2
            + (self.rho * D**2 * self.CT2) * Va**2
        )
        Q = ((self.rho * D**5 * self.CQ0) /(4*np.pi**2) * rpm**2 
            + (self.rho * D**4 * self.CQ1)/(2*np.pi) * Va * rpm**2
            + (self.rho * D**3 * self.CQ2) * Va**2
        )
        rotor_force = rotor_R_body @ np.diag(T.ravel())
        rotor_moment = rotor_R_body @ np.diag(Q.ravel()) + np.cross(self.rotor_pos, rotor_force, axisa=0, axisb=0, axisc=0)
        
        # Sum along second axis to get total force/moment
        return rotor_force.sum(axis=-1, keepdims=True), rotor_moment.sum(axis=-1, keepdims=True)

    def __call__(self, t, y, u):
        """ODE function
        
        If vectorized, shape is (n,k), otherwise it's (n,)
        For now, assumed not vectorized    
        """

        dydt = np.zeros(self.n)
        y = y.reshape(-1,1)
        u = u.reshape(-1,1)

        """Unpack"""
        # State
        pos = y[0:3]
        vel = y[3:6]
        R = y[6:15].reshape(3,3)
        pqr = y[15:18]
        tet_r = y[18:21]

        # Control
        dE = u[0:2]
        rpm = u[2:5]
        tet_c = u[5:8]
        
        omega_skew = skew(pqr)

        """Aerodynamics"""
        v_wind = np.zeros((3,1))
        v_a = vel - R.T @ v_wind
        F_aero, M_aero = self.aero_forces_and_moments(v_a, pqr, dE)

        """Rotor"""
        F_rotor, M_rotor = self.rotor_forces_and_moments(v_a, tet_c, rpm)

        """Compute Derivatives"""
        # Position
        dydt[0:3] = (R @ vel).ravel()

        # Velocity
        dydt[3:6] = (
            omega_skew @ vel + self.g*R.T@self.e3 + (F_aero + F_rotor) / self.mass
        ).ravel()

        # Rotation
        dydt[6:15] = (R @ omega_skew).ravel()

        # Body Rate
        dydt[15:18] = (
            -self.J_inv @ omega_skew @ self.J @ pqr + self.J_inv @ (M_aero + M_rotor)
        ).ravel()

        # Tilt Servo
        dydt[18:21] = (self.k_tilt * (tet_c - tet_r)).ravel()
        
        return dydt

    
if __name__ == "__main__":
    dynamics = QuadVTOL()

    x0 = np.array([
        0,0,0, 0,0,0, 1,0,0, 0,1,0, 0,0,1, 0,0,0, 0,0,0
    ])
    u = np.array([
        0,0, 1,1,1, 0,0,0
    ])
    
    sol = solve_ivp(
        partial(dynamics, u=u),
        t_span=[0,10],
        y0=x0
    )
    t = sol.t
    y = sol.y

    pos = y[0:3]
    vel = y[3:6]
    # R = y[6:15].reshape(3,3)
    pqr = y[15:18]
    # tet_r = y[18:21]

    print(f"t:{t.shape}, y:{y.shape}")

    f, ax = plt.subplots(3,1, sharex=True)
    ax[0].plot(t,vel[0])
    ax[1].plot(t,vel[1])
    ax[2].plot(t,vel[2])

    plt.show()

