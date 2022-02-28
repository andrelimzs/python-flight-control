import numpy as np
from Utils import *

class Translation:
    num_states = 6
    x0 = np.zeros(6)
    def __call__(self, params, T, LMN, x):
        # Form thrust vector (in body frame)
        thrust = np.concatenate([0*T, 0*T, -T])

        # Position
        d_pos = x['vel']

        # Velocity
        d_vel = apply(x['eRb'], thrust) / params['mass']
        d_vel[2] += 9.81

        return np.concatenate([d_pos, d_vel])


class EulerRotation:
    num_states = 3
    x0 = np.zeros(6)
    def body_rate_to_euler_dot(self, eul):
        """ Compute the transformation to go from body rates pqr to euler derivative
        for either a vector (3,) -> (3,3) or a sequence of vectors (3,N) -> (N,3,3) """
        # Align eul to (3,N)
        if eul.shape[0] != 3:
            eul = eul.T

        # Get length N
        N = eul.shape[1] if eul.ndim > 1 else 1

        # Precompute the sin and cos terms
        one = np.ones_like(eul[0]).reshape(-1,1,1)
        zero = np.zeros_like(eul[0]).reshape(-1,1,1)
        s2 = np.sin(eul[2]).reshape(-1,1,1)
        c2 = np.cos(eul[2]).reshape(-1,1,1)
        t1 = np.tan(eul[1]).reshape(-1,1,1)
        sec1 = 1 / np.cos(eul[1]).reshape(-1,1,1)

        # Form rotation matrix (N,3,3)
        M = np.block([ [one,  s2*t1,   c2*t1],
                       [zero,   c2,     -s2],
                       [zero, s2*sec1, c2*sec1]
                     ])
        return np.squeeze(M)

    def __call__(self, params, T, LMN, x):
        # Convert body-axis rates pqr to euler derivative
        eul_dot = apply(self.body_rate_to_euler_dot(x['att']), x['rate'])
        
        # Attitude
        d_att = eul_dot

        # Angular Rate
        euler_cross_product = apply(skew(x['rate']) @ params['J'], x['rate'])
        d_rate = 1 / params['J'] * (LMN - euler_cross_product)
        
        return np.concatenate([d_att, d_rate])


class QuaternionRotation:
    num_states = 4
    x0 = np.array([1.,0.,0.,0.,0.,0.,0.])
    def __init__(self, K=0.1):
        # Renormalisation gain
        self.K = K
        
    def make_Omega(self, omega):
        w1 = omega[0].reshape(-1,1,1)
        w2 = omega[1].reshape(-1,1,1)
        w3 = omega[2].reshape(-1,1,1)
        z  = np.zeros_like(w1)
        Omega = np.block([ [z,  -w1, -w2, -w3],
                           [w1,  z,   w3, -w2],
                           [w2, -w3,  z,   w1],
                           [w3,  w2, -w1,  z ] ])
        return np.squeeze(Omega)
        
    def __call__(self, params, T, LMN, x):
        q = x['att']
        
        # Use quaternion derivative formulation of \dot{q} = 0.5 * Omega * q
        Omega = self.make_Omega(x['rate'])
        d_quat = 0.5 * apply(Omega, q)
        
        # Gain to limit norm drift
        c = self.K * (np.ones_like(q) - dot_2d(q,q))
        d_quat += c * q
        
        # Angular Rate
        euler_cross_product = apply(skew(x['rate']) @ params['J'], x['rate'])
        d_rate = 1 / params['J'] * (LMN - euler_cross_product)
        
        return np.concatenate([d_quat, d_rate])


class DefaultRotor:
    num_states = 0
    x0 = np.zeros(0)
    def __init__(self, T_max, LMN_max):
        self.T_max = T_max
        self.LMN_max = LMN_max
        
    def __call__(self, params, u, x):
        T = np.atleast_1d(u[0])
        LMN = u[1:4]
        
        # Saturate the Thrust and Torques
        T = np.clip(T, 0, self.T_max)
        LMN = np.clip(LMN, -self.LMN_max, self.LMN_max)
        
        return T, LMN


class DefaultAero:
    num_states = 0
    x0 = np.zeros(0)
    def __call__(self, params, u, x):
        d_pos = np.zeros_like(x['pos'])
        d_vel = np.zeros_like(x['vel'])
        d_att = np.zeros_like(x['att'])
        d_rate = np.zeros_like(x['rate'])
        return np.concatenate([d_pos, d_vel, d_att, d_rate])


class Quadcopter:
    def __init__(self, rotationType='Euler', params=None, reference=None, control=None, rotor=None, aero=None):
        # Assign values, with defaults
        self.reference = reference if reference else DefaultReference
        self.control = control if control else StateFeedbackControl
        self.rotor = rotor if rotor else DefaultRotor(30,1)
        self.aero = aero if aero else DefaultAero()
        self.params = params if params else DefaultParams()
        
        self.translation = Translation()
        # Choose rotational dynamics
        if rotationType == 'Quaternion':
            self.rotation = QuaternionRotation()
            self.att2rotm = quat2rotm
            self.att2eul  = quat2eul
        else:
            self.rotation = EulerRotation()
            self.att2rotm = eul2rotm
            self.att2eul  = lambda x: x
        
        # Calculate the index to unpack each state
        self.state_vec = {'pos' : range(0,3),
                          'vel' : range(3,6)}
        i = 6; j = i + self.rotation.num_states
        self.state_vec['att']   = range(i,j)
        i = j; j = i + 3
        self.state_vec['rate']   = range(i,j)
        i = j; j = i + self.rotor.num_states
        self.state_vec['rotor'] = range(i,j)
        i = j; j = i + self.aero.num_states
        self.state_vec['aero']  = range(i,j)
        
    def unpack_state(self, y):
        """Unpack y (np.ndarray) into state (dict)"""
        y = np.squeeze(y)
        states = ['pos','vel','att','rate','rotor','aero']
        x = { s : y[self.state_vec[s]] for s in states }
        
        # Compute certain useful values
        x['eRb'] = self.att2rotm(x['att'])
        x['eul'] = self.att2eul(x['att'])
        
        return x
        
    def generate_x0(self):
        """Generate an initial state vector"""
        x0 = np.concatenate([ self.translation.x0,
                              self.rotation.x0,
                              self.rotor.x0,
                              self.aero.x0 ])
        return x0
        
    def __call__(self, t, y):
        """ODE function"""
        state = self.unpack_state(y)
        
        # Execute trajectory generation and control law
        ref = self.reference(t)
        u = self.control(ref, state)
        
        # Simulate rotor, translation and rotation
        T,LMN = self.rotor(self.params, u, state)
        d_translation = self.translation(self.params, T, LMN, state)
        d_rotation = self.rotation(self.params, T, LMN, state)
        dydt = np.concatenate([d_translation, d_rotation])
        
        # Simulate Aerodynamics
        dydt += self.aero(self.params, u, state)
        
        return dydt