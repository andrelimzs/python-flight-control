# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python [conda env:control-systems] *
#     language: python
#     name: conda-env-control-systems-py
# ---

# +
import numpy as np
import numpy.linalg as LA
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import control

import time
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

# Create a new visualizer
vis = meshcat.Visualizer()

# +
from scipy.spatial.transform import Rotation as R

def rotx(a):
    """Rotation matrix about x"""
    return R.from_euler('x', -a).as_matrix()

def roty(a):
    """Rotation matrix about y"""
    return R.from_euler('y', -a).as_matrix()

def rotz(a):
    """Rotation matrix about z"""
    return R.from_euler('z', -a).as_matrix()


def eul2rotm(eul) -> np.ndarray:
    """ Rotation matrix from euler angles
    If input is (3,) return (3,3)
    If input is (3,N) return (N,3,3) """
    # Align eul to (3,N)
    if eul.shape[0] != 3:
        eul = eul.T
    
    # Get length N
    N = eul.shape[1] if eul.ndim > 1 else 1
    
    # Precompute the sin and cos terms
    s = []
    c = []
    for i in range(3):
        s.append( np.sin(eul[i]).reshape(-1,1,1) )
        c.append( np.cos(eul[i]).reshape(-1,1,1) )
    
    # Form rotation matrix (N,3,3)
    M = np.block([ [c[2]*c[1], c[2]*s[1]*s[0] - s[2]*c[0], c[2]*s[1]*c[0] + s[2]*s[0]],
                   [s[2]*c[1], s[2]*s[1]*s[0] + c[2]*c[0], s[2]*s[1]*c[0] - c[2]*s[0]],
                   [ -s[1],               c[1]*s[0],                  c[1]*c[0]      ] ])
    return M

def apply(M, vec) -> np.ndarray:
    """ Apply a sequence of linear transformations (Nx3x3)
    to either a vector (3,) or a sequence of vectors (3,N) """
    # Check for the single Rot matrix case (3,3) and return instantly
    if M.ndim < 3:
        return M @ vec
    
    # Apply N rotations to a single vector
    if vec.ndim == 1:
        for R in M:
            return R @ vec
        
    # Apply N rotations to N vectors
    elif M.shape[0] == vec.shape[1]:
        N = M.shape[0]
        output = np.zeros((3,N))
        for i,R in enumerate(M):
            output[:,i] = R @ vec[:,i]
        return output
    
    # N rotations cannot be applied to M vectors
    else:
        raise ValueError("Cannot apply different number of transformations and vectors")

def skew(vec):
    """ Form the skew-symmetric matrix
    to either a vector (3,) -> (3,3) or a sequence of vectors (3,N) -> (N,3,3) """
    a1 = vec[0].reshape(-1,1,1)
    a2 = vec[1].reshape(-1,1,1)
    a3 = vec[2].reshape(-1,1,1)
    zero = np.zeros_like(a1)
    M = np.block( [[zero, -a3,    a2],
                   [ a3,  zero,  -a1],
                   [-a2,   a2,   zero]] )
    return M
    
def atan2(Y, X) -> np.ndarray:
    """Numpy's arctan2, but output is at least 1D array"""
    return np.atleast_1d(np.arctan2(Y,X))

def stack_squeeze(arr) -> np.ndarray:
    """Stack along axis 0, then squeeze to remove any trailing dimensions of size 1"""
    return np.squeeze(np.stack( arr ))


# -

# **Setup**
# 1. Multicopter physics
# 2. Generate trajectories
# 3. Define control law

# +
class Translation:
    num_states = 6
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
        return M

    def __call__(self, params, T, LMN, x):
        # Convert body-axis rates pqr to euler derivative
        eul_dot = apply(self.body_rate_to_euler_dot(x['att']), x['rate'])
        
        # Attitude
        d_att = eul_dot

        # Angular Rate
        euler_cross_product = apply(skew(x['rate']) @ params['J'], x['rate'])
        d_rate = 1 / params['J'] * (LMN - euler_cross_product)
        
        return np.concatenate([d_att,d_rate])
    
class DefaultRotor:
    num_states = 0
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
    def __call__(self, params, u, x):
        d_pos = np.zeros_like(x['pos'])
        d_vel = np.zeros_like(x['vel'])
        d_att = np.zeros_like(x['att'])
        d_rate = np.zeros_like(x['rate'])
        return np.concatenate([d_pos, d_vel, d_att, d_rate])


# -

class Quadcopter(object):
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
        else:
            self.rotation = EulerRotation()
            self.att2rotm = eul2rotm
        
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
        y = np.squeeze(y)
        states = ['pos','vel','att','rate','rotor','aero']
        x = { s : y[self.state_vec[s]] for s in states }
        
        # Compute certain useful values
        x['eRb']  = self.att2rotm(x['att'])
        
        return x
    
    def __call__(self, t, y):
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


# + tags=[]
# Compute state feedback
z3 = np.zeros((3,3))
I3 = np.eye(3)
A = np.block([[z3, I3],[z3, z3]])
B = np.block([[z3],[I3]])
p = np.repeat([-3,-4], 3)
K_pos = control.place(A, B, p)

# Compute state feedback
z3 = np.zeros((3,3))
I3 = np.eye(3)
A = np.block([[z3, I3],[z3, z3]])
B = np.block([[z3],[I3]])
p = np.repeat([-10,-12], 3)
K_att = control.place(A, B, p)

# Define Control Law
def state_feedback(ref,x):
    """
    Nested control structure
    1. Position + Velocity
    1b Convert Acceleration -> Orientation + Thurst
    2. Attitude (Orientation) + Angular Rate
    
    Coordinate Frames
    - Position and Velocity are in NED
    """
    # Convert x to np.ndarray if necessary
    # if type(x) is not np.ndarray: x = np.array(x)
    ref = ref.copy()
    
    #[DEBUG] print(f'x: {x.shape} \t ref: {ref.shape}')
    
    # Physical parameters
    mass = 1
    MOI = np.diag([0.01, 0.01, 0.02])
    
    # Unpack state
    if type(x) is np.ndarray:
        pos  = x[0:3]
        vel  = x[3:6]
        att  = x[6:9]
        rate = x[9:12]
        
    elif type(x) is dict:
        pos  = x['pos']
        vel  = x['vel']
        att  = x['att']
        rate = x['rate']
    
    # Unpack reference
    ref_pos  = ref[0:3]
    ref_vel  = ref[3:6]
    ref_att  = ref[6:9]
    ref_rate = ref[9:12]
    
    # Disable control loop if reference command all outer-loop commands are nan
    disable_pos = np.isnan(ref_pos)
    disable_vel = np.isnan(ref_vel) & disable_pos
    disable_att = np.isnan(ref_att) & disable_vel
    
    #[DEBUG] print(f'disable_pos: {disable_pos.shape} \t disable_vel: {disable_vel.shape} \t disable_att: {disable_att.shape}')
    #[DEBUG] print(f'ref_pos: {ref_pos.shape} \t pos: {pos.shape}')
    
    ref_pos[disable_pos] = pos[disable_pos]
    ref_vel[disable_vel] = vel[disable_vel]
    ref_att[disable_att] = att[disable_att]
    
    # Replace the remaining NaNs with zero
    ref_vel[np.isnan(ref_vel) & ~disable_vel] = 0
    ref_att[np.isnan(ref_att) & ~disable_att] = 0
    ref_rate[np.isnan(ref_rate)] = 0
    
    # Check for NaNs
    if np.isnan(pos).any():
        raise ValueError('pos is nan')
    if np.isnan(vel).any():
        raise ValueError('vel is nan')
    if np.isnan(att).any():
        raise ValueError('att is nan')
    if np.isnan(rate).any():
        raise ValueError('rate is nan')
        
    if np.isnan(ref_vel).any():
        raise ValueError('ref_vel is nan')
    if np.isnan(ref_att).any():
        raise ValueError('ref_att is nan')
    if np.isnan(ref_rate).any():
        raise ValueError('ref_rate is nan')
    
    # Position + Velocity Subsystem
    pos_err = ref_pos - pos
    vel_err = ref_vel - vel
    acc_des = K_pos @ np.concatenate([pos_err, vel_err])
    
    # Add gravity
    acc_des[2] -= 9.81
    
    acc_des[2] = np.clip(acc_des[2], -20, -5)
    
    # # Saturate a_des while preserving direction
    # z_des_violation = acc_des[2,:] - np.clip(acc_des[2,:], -20, -5)   
    
    # Convert acceleration into orientation + thrust
    # Transform pos/vel from NED frame to body frame 
    acc_des_b = apply( rotz(-att[2]), acc_des )
    
    # Compute desired roll/pitch from desired acceleration
    T_des = np.atleast_1d(LA.norm(acc_des_b, axis=0)) * mass
    # T_des = -acc_des_b[2,:].reshape(1,-1) * mass
    
    # Enforce T_des shape to either (1,) or (1,N)
    if T_des.shape[0] > 1:
        T_des = T_des.reshape((1,-1))

    phi_des   =  np.arcsin(acc_des_b[1] / T_des)
    theta_des = -atan2(acc_des_b[0], -acc_des_b[2])
    psi_des   =  np.zeros(phi_des.shape)
    
    if theta_des.shape[0] > 1:
        theta_des = theta_des.reshape((1,-1)) 
    
    #[DEBUG] print(f'acc_des_b: {acc_des_b.shape}')
    #[DEBUG] print(f'phi_des: {phi_des.shape} \t theta_des: {theta_des.shape} \t psi_des: {psi_des.shape}')
    
    # Form att desired vector
    att_des = stack_squeeze([phi_des, theta_des, psi_des])
    
    if np.isnan(att_des).any():
        print(acc_des_b[1] / T_des)
        raise ValueError(f'att_des {att_des} is nan')
        
    #[DEBUG] print(f'ref_att: {ref_att.shape} \t att_des: {att_des.shape} \t att: {att.shape}')
    
    # Attitude Subsystem
    att_err  = ref_att + att_des - att 
    rate_err = ref_rate - rate
    
    #[DEBUG] print(f'att_err: {att_err.shape} \t rate_err: {rate_err.shape}')
        
    ang_acc_des  = K_att @ np.concatenate([att_err, rate_err])
    LMN_des = MOI @ ang_acc_des
    
    #[DEBUG] print(f'MOI: {MOI.shape} \t ang_acc_des: {ang_acc_des.shape}')
    #[DEBUG] print(f'T_des: {T_des.shape} \t LMN_des: {LMN_des.shape}')
    
    control_u = np.concatenate([T_des, LMN_des])
    
    return control_u


# +
# %%time
# Generate trajectories
def ref_function(t):
    t = np.atleast_1d(t)
    N = t.shape[0]
    
    zero = 0 * t
    nan  = np.nan * t
    step = -1 * (t > 0).astype(int)
    
    sin1 = 4 * np.sin(1*t)
    sin2 = 4 * np.sin(2*t)
    sin3 = 1 * np.sin(0.5*t) - 2
    
    square1 = 4 * (np.sin(1*t) > 0).astype(int)
    square2 = 4 * (np.sin(1*t) < 0).astype(int)
    
    ref_pos  = stack_squeeze( [sin1, sin2, step] )
    ref_vel  = stack_squeeze( [nan, nan, nan] )
    ref_att  = stack_squeeze( [nan, nan, zero] )
    ref_rate = stack_squeeze( [nan, nan, nan] )
    
    return np.concatenate( [ref_pos, ref_vel, ref_att, ref_rate] )

# Run Simulation
Ts = 0.01
t_span = [0,20]
t_eval = np.arange(t_span[0], t_span[1], Ts)
y0 = np.zeros(12)
# sol = solve_ivp(eqn_of_motion, t_span, y0, t_eval=t_eval,
#                 args=(state_feedback, ref_function))

params = { 'mass': 1, 'J': np.array([0.01, 0.01, 0.02])}
quadcopter = Quadcopter(params=params, reference=ref_function, control=state_feedback)
sol = solve_ivp(quadcopter, t_span, y0, t_eval=t_eval)

# Unpack solution
t = sol.t
pos  = sol.y[0:3]
vel  = sol.y[3:6]
att  = sol.y[6:9]
rate = sol.y[9:12]

# Reference
ref  = ref_function(sol.t)
ref_pos  = ref[0:3]
ref_vel  = ref[3:6]
ref_att  = ref[6:9]
ref_rate = ref[9:12]

# Control input
control_u = state_feedback(ref, sol.y)
T   = control_u[0]
LMN = control_u[1:4]

# +
# %matplotlib inline
plt.rcParams["figure.figsize"] = (20,5)

fig,ax = plt.subplots(3,2, sharex=True)
for i in range(3):
    ax[i,0].plot(t, pos[i])
    ax[i,0].plot(t, ref_pos[i])
    
    ax[i,1].plot(t, vel[i])
    ax[i,1].plot(t, ref_vel[i])

[ ax[i,j].grid() for i in range(3) for j in range(2) ]
ax[0,0].set_title('Pos')
ax[0,1].set_title('Vel')
        
fig,ax = plt.subplots(3,2, sharex=True)
for i in range(3):
    ax[i,0].plot(t, np.rad2deg(att[i]))
    ax[i,0].plot(t, np.rad2deg(ref_att[i]))
    
    ax[i,1].plot(t, np.rad2deg(rate[i]))
    ax[i,1].plot(t, np.rad2deg(ref_rate[i]))
    # ax[i,3].plot(time, LMN[i])
    
[ ax[i,j].grid() for i in range(3) for j in range(2) ]
ax[0,0].set_title('Att')
ax[0,1].set_title('Rate')


# -

# **Visualise**

# + tags=[]
def create_quadcopter(vis):
    arm_length = 0.25
    rotor_radius = 0.125

    # Arms
    vis['drone']['left_arm'].set_object(g.Box([2*arm_length, 0.05, 0.05]))
    vis['drone']['left_arm'].set_transform(tf.rotation_matrix(np.deg2rad(45), [0,0,1]))
    vis['drone']['right_arm'].set_object(g.Box([2*arm_length, 0.05, 0.05]))
    vis['drone']['right_arm'].set_transform(tf.rotation_matrix(np.deg2rad(-45), [0,0,1]))

    # Rotors
    for i in range(1,5):
        theta = np.deg2rad(45 + 90*i)
        offset = np.array([arm_length * np.sin(theta), arm_length * np.cos(theta), 0.05])

        # Compute transformation
        T = tf.rotation_matrix(np.deg2rad(90), [1,0,0])
        T[0:3,3] = offset

        vis['drone'][f'rotor{i}'].set_object(g.Cylinder(0.01, rotor_radius))
        vis['drone'][f'rotor{i}'].set_transform(T)
        
    return vis['drone']

vis.delete()
drone = create_quadcopter(vis)
vis.jupyter_cell()
# -

for i in range(ref.shape[1]):
    # Convert NED to ENU
    pos_enu = np.array([1,-1,-1]) * pos[:,i]
    att_enu = np.array([1,-1,-1]) * att[:,i]
    
    # Form homogeneous transformation
    T = tf.euler_matrix(*att_enu)
    T[0:3,3] = pos_enu
    
    # Apply
    vis.set_transform(T)
    time.sleep(0.001)


