import numpy as np
import numpy.linalg as LA
from flightcontrol.Utils import *
import matplotlib.pyplot as plt
import control
from copy import deepcopy

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
        self.n = 16
        # Physical parameters
        self.g = 9.81 # [m/s^2]
        self.e3 = np.array([0.,0.,1.])
        self.mass = 0.771 # [kg]
        self.J = np.array([
            [0.0165,    0.0,    4.8e-5], 
            [0.0,       0.0128, 0.0   ],
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
        alpha = atan2(v_a[2], v_a[0])
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
        ])

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
        ])

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
            [np.cos(tet_r[0]),  np.cos(tet_r[1]),    0],
            [0,                 0,                   0],
            [-np.sin(tet_r[0]), -np.sin(tet_r[1]),  -1]
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
        return rotor_force.sum(axis=-1), rotor_moment.sum(axis=-1)

    def __call__(self, t, y, u):
        """ODE function
        
        If vectorized, shape is (n,k), otherwise it's (n,)
        For now, assumed not vectorized    
        """
        if not np.isfinite(y).all():
            raise ValueError("State not finite")

        if not np.isfinite(u).all():
            raise ValueError("Control input not finite")
        
        dydt = np.zeros_like(y)

        y = deepcopy(y)

        """Unpack"""
        # State
        pos = y[0:3]
        vel = y[3:6]
        # R = y[6:15].reshape(3,3)
        quat = y[6:10]
        pqr = y[10:13]
        tet_r = y[13:15]

        # Control
        dE = u[0:2]
        # rpm = u[2:5]
        # tet_c = u[5:7]
        rpm = np.array([0., 0., 0.])
        tet_c = np.array([0., 0.])
        TLMN = u[2:6]

        # Convert quaternion to rotation matrix
        R = quat2rotm(quat)

        # Check that R is orthogonal
        if np.abs(LA.det(R) - 1) > 1e-5:
            raise ValueError(f"Rotation matrix det = {LA.det(R)}")
        
        omega_skew = skew(pqr)

        """Aerodynamics"""
        v_wind = np.zeros((3,))
        v_a = vel - R.T @ v_wind
        F_aero, M_aero = self.aero_forces_and_moments(v_a, pqr, dE)

        """Rotor"""
        F_rotor, M_rotor = self.rotor_forces_and_moments(v_a, tet_r, rpm)

        # DEBUG Use TLMN directly
        M_rotor = TLMN[1:4]
        F_rotor = np.array([0., 0., -TLMN[0]])

        # DEBUG Disable aerodynamics
        if 0:
            F_aero *= 0
            M_aero *= 0

        """Compute Derivatives"""
        # Position
        dydt[0:3] = R @ vel

        # Velocity
        dydt[3:6] = omega_skew @ vel + self.g*R.T@self.e3 + (F_aero + F_rotor) / self.mass

        # Rotation
        w1, w2, w3 = pqr
        Omega = np.block([
            [0,  -w1, -w2, -w3],
            [w1,  0,   w3, -w2],
            [w2, -w3,  0,   w1],
            [w3,  w2, -w1,  0 ]
        ])
        dydt[6:10] = 0.5 * Omega @ quat

        # Quaternion renormalization gain
        quat_K = 0.1
        dydt[6:10] += quat_K * (1.0 - np.dot(quat,quat)) * quat

        # Body Rate
        dydt[10:13] = self.J_inv@(M_aero + M_rotor - omega_skew @ self.J @ pqr)

        # Tilt Servo
        dydt[13:15] = self.k_tilt * (tet_c - tet_r)
        
        return dydt

def quatKronecker(quat):
    q0,q1,q2,q3 = quat
    return np.array([
        [q0, -q1, -q2, -q3],
        [q1,  q0, -q3,  q2],
        [q2,  q3,  q0, -q1],
        [q3, -q2,  q1,  q0]
    ])

def quatConjugate(quat):
    quat[1:4] *= -1
    return quat

def quatInv(quat):
    return quatConjugate(quat) / LA.norm(quat)

class PID():
    def __init__(self, dt):
        self.dt = dt

        rotor_pos = np.array([
            [0.1, 0.1, -0.22],
            [0.18, 0.18, 0],
            [0, 0, 0]
        ])
        x1,x2,x3 = rotor_pos[0]
        y1,y2,_ = rotor_pos[1]
        self.m = m = 0.771
        g = 9.81
        self.J = np.array([
            [0.0165,    0.0,    4.8e-5], 
            [0.0,       0.0128, 0.0   ],
            [4.8e-5,    0.0,    0.0282]
        ])
        M = np.array([
            [ 1,   1,    1,   0,        0       ],
            [ x1,  x2,  -x3,  0,        0       ],
            [-y1,  y2,   0,   0,        0       ],
            [ 0,   0,    0,  -y1*m*g/3, y2*m*g/3]
        ])
        self.M_inv = LA.pinv(M)

        # Integrators
        self.quat_err_int = np.zeros(3)
        self.pqr_err_int = np.zeros(3)

        # Compute state feedback
        z3 = np.zeros((3,3))
        I3 = np.eye(3)
        A = np.block([[z3, I3],[z3, z3]])
        B = np.block([[z3],[I3]])
        p = np.repeat([-3,-4], 3)
        self.K_rot = control.place(A, B, p)

    def acc_to_att(self, acc_des_b):
        # Compute desired roll/pitch from desired acceleration
        T_des = LA.norm(acc_des_b, axis=0) * self.m
        # T_des = -acc_des_b[2,:].reshape(1,-1) * mass
        
        phi_des   =  np.arcsin(acc_des_b[1] / T_des)
        theta_des = -atan2(acc_des_b[0], -acc_des_b[2])
        psi_des   =  np.zeros(phi_des.shape)

        # Form att desired vector
        att_des = np.squeeze(np.stack([phi_des, theta_des, psi_des]))
        
        return T_des, att_des
    
    def run(self, ref, x):
        if not np.isfinite(x).all():
            raise ValueError("State not finite")
        
        m = 0.771
        g = 9.81

        # [!] Deepcopy state to prevent overriding ODE solver solution
        x = deepcopy(x)

        # Unpack state
        pos = x[0:3]
        vel = x[3:6]
        # R = x[6:15].reshape(3,3)
        quat = x[6:10]
        pqr = x[10:13]
        tet_r = x[13:15]

        # Convert quaternion to rotation matrix
        R = quat2rotm(quat)

        # Unpack reference
        vel_des = ref[0:3]
        psi_des = ref[3]

        # Velocity Loop (Body frame)
        velP = 1
        vel_err = vel_des - vel
        acc_des = velP * vel_err

        # Convert acceleration to desired thrust vector (inertial frame)
        acc_des_I = R @ acc_des - np.array([0, 0, m*g])
        T_des = LA.norm(acc_des_I)
        
        # Construct desired rotation matrix from thrust vector
        x_des1 = np.array([cos(psi_des), sin(psi_des), 0.0])
        z_des = -acc_des_I
        y_des = np.cross(z_des, x_des1)
        x_des = np.cross(y_des, z_des)

        R_des = np.stack([x_des, y_des, z_des], axis=1)

        """SE(3) Error"""
        # Convert acceleration to desired thrust vector (inertial frame)
        acc_des_I = R @ acc_des - np.array([0, 0, m*g])
        T_des = LA.norm(acc_des_I)
        
        # Construct desired rotation matrix from thrust vector
        x_des1 = np.array([cos(psi_des), sin(psi_des), 0.0])
        z_des = -acc_des_I
        y_des = np.cross(z_des, x_des1)
        x_des = np.cross(y_des, z_des)

        R_des = np.stack([x_des, y_des, z_des], axis=1)

        """Rotation Matrix"""
        if 0:
            # From Geometric Tracking Control of a Quadrotor UAV on SE(3)
            pqr_des = np.array([0., 0., 0.])
            e_R = 0.5 * veemap(R_des.T @ R - R.T @ R_des)
            e_Om = pqr - R.T @ R_des @ pqr_des

            kR = 10.0
            kV = 2.0
            M_des = kR * e_R + kV * e_Om + np.cross(pqr, self.J @ pqr)
            # - self.J @ hatmap(pqr)@R.T@pqr_des # No pqr command

        """Quaternion Error"""
        if 1:
            quat_des = rotm2quat(R_des)
            # # TEST Override ref cmd
            # quat_des = eul2quat(np.array([0.0, -0.7, 0.0]))

            # Attitude
            # quat_err = quatKronecker(quat_des) @ quatConjugate(quat)
            quat_err = quatKronecker(quatInv(quat)) @ quat_des
            if quat_err[0] < 0:
                quat_err *= -1
            self.quat_err_int += self.dt * quat_err[1:4]

            KP_quat = np.array([6.5, 6.5, 6.5]) * 1
            KI_quat = np.array([6.5, 6.5, 6.5]) * 0
            pqr_des = KP_quat * quat_err[1:4] + KI_quat * self.quat_err_int

            # Clamp pqr_des
            # Limit to 220 deg/s (PX4 limit)
            pqr_des = pqr_des.clip(min=-3.8, max=3.8)

            # Rate
            pqr_err = pqr_des - pqr
            self.pqr_err_int += self.dt * pqr_err
            KP_pqr = np.array([0.15, 0.15, 0.15]) * 100
            KI_pqr = np.array([0.2, 0.2, 0.2]) * 100
            ang_acc_des = KP_pqr * pqr_err + KI_pqr * self.pqr_err_int

            ang_acc_des = ang_acc_des.clip(min=-100, max=100)
            
            # Convert angular acceleration to torque
            LMN_des = self.J @ ang_acc_des + np.cross(pqr, self.J@pqr)

            # Logging
            eul_des = quat2eul(quat_des)

        """Euler Error"""
        if 0:
            acc_des_b = acc_des - R.T @ np.array([0,0,m*g])
            T_des, eul_des = self.acc_to_att(acc_des_b)

            eul_des = np.array([0.0, -1.0, 0.0])
            pqr_des = np.ones(3) * np.nan

            eul = quat2eul(quat)
            eul_err = eul_des - eul
            pqr_err = -pqr

            ang_acc_des = self.K_rot @ np.concatenate([eul_err, pqr_err])
            LMN_des = self.J @ ang_acc_des + np.cross(pqr, self.J@pqr)

        # Convert TLMN into RPM
        TLMN_des = np.concatenate([T_des.reshape(1), LMN_des], axis=0)
        u = self.M_inv @ TLMN_des

        dE = np.array([0.,0.])
        # return np.concatenate([dE, u])
        return np.concatenate([dE, TLMN_des]), (eul_des, pqr_des)

