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

# def eul2rotm(eul):
#     """Rotation matrix from euler angles"""
#     return R.from_euler('zyx', -eul.T).as_matrix()

def eul2rotm(eul):
    """Rotation matrix from euler angles"""
    eul = add_trailing_dim(eul)
    
    # Align eul to (3,N)
    if eul.shape[0] != 3:
        eul = eul.copy().T
    
    N = eul.shape[1]
    
    # Precompute the sin and cos terms
    s = []
    c = []
    for i in range(3):
        s.append( np.sin(eul[i,:]).reshape(-1,1,1) )
        c.append( np.cos(eul[i,:]).reshape(-1,1,1) )
    
    # Form rotation matrix (N,3,3)
    M = np.block([ [c[2]*c[1], c[2]*s[1]*s[0] - s[2]*c[0], c[2]*s[1]*c[0] + s[2]*s[0]],
                   [s[2]*c[1], s[2]*s[1]*s[0] + c[2]*c[0], s[2]*s[1]*c[0] - c[2]*s[0]],
                   [ -s[1],               c[1]*s[0],                  c[1]*c[0]      ] ])
    return M

def apply_rot(rot_matrix, vec):
    """
    Apply a sequence of rotation matrices (Nx3x3)
    to either:
    - a vector (3,)
    - a sequence of vector (3,N)
    """
    # Check for the single R case (3,3) instead of (Nx3x3)
    if rot_matrix.ndim < 3:
        return rot_matrix @ vec
    
    output = []
    # Apply R to a single vector or N vectors
    if vec.ndim == 1:
        for R in rot_matrix:
            output.append(R @ vec)
    else:
        for i,R in enumerate(rot_matrix):
            output.append(R @ vec[:,i])
    
    return np.stack(output, axis=1)

def add_trailing_dim(x):
    if x.ndim <= 1:
        x = x.reshape((-1,1))
    return x


# -

# **Setup**
# 1. Multicopter physics
# 2. Generate trajectories
# 3. Define control law

# + tags=[]
# Define Multicopter Physics
def eqn_of_motion(t, y, control_law, ref_function):
    mass = 1
    J_inv = np.array([1/0.01, 1/0.01, 1/0.02]).reshape(3,1)
    T_max = 30
    LMN_max = 1
    
    # Unpack state
    y = add_trailing_dim(y)
    pos = y[0:3,:]
    vel = y[3:6,:]
    att = y[6:9,:]
    rate = y[9:12,:]
    
    eRb = eul2rotm(att)
    
    # Call trajectory and control functions
    ref = ref_function(t)
    control_u = control_law(ref, y)
    
    # ================  Actuator Model  =================
    T = control_u[0,:].reshape(1,-1)
    LMN    = control_u[1:4,:].reshape(3,-1)
    
    T = np.clip(T, 0, T_max)
    LMN = np.clip(LMN, -LMN_max, LMN_max)
    
    thrust = np.concatenate([np.zeros(T.shape), np.zeros(T.shape), -T])
    
    # ===================  Calculate  ===================
    # pass
    
    # ==============  Compute derivatives  ==============
    # State is { pos, vel, att, rate }
    
    # Position
    d_pos = vel
    
    # Velocity
    d_vel = apply_rot(eRb, thrust) / mass
    
    # Gravity
    d_vel[2] += 9.81
    
    # Attitude
    d_att = rate
    
    # Angular Rate
    d_rate = J_inv * LMN
    
    dydt = np.concatenate([d_pos, d_vel, d_att, d_rate], axis=0)
    
    return np.squeeze(dydt)


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
    if type(x) is not np.ndarray: x = np.array(x)
    # Add trailing dimension
    x = add_trailing_dim(x)
    ref = add_trailing_dim(ref.copy())
    
    # Physical parameters
    mass = 1
    MOI = np.array([0.01, 0.01, 0.02]).reshape((3,1))
    
    # Unpack state
    pos  = x[0:3,:]
    vel  = x[3:6,:]
    att  = x[6:9,:]
    rate = x[9:12,:]
    
    # Unpack reference
    ref_pos  = ref[0:3,:]
    ref_vel  = ref[3:6,:]
    ref_att  = ref[6:9,:]
    ref_rate = ref[9:12,:]
    
    # Disable control loop if reference command all outer-loop commands are nan
    disable_pos = np.isnan(ref_pos)
    disable_vel = np.isnan(ref_vel) & disable_pos
    disable_att = np.isnan(ref_att) & disable_vel
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
    acc_des[2,:] -= 9.81
    # Clip acc_des
    acc_des[2,:] = np.clip(acc_des[2,:], -20, -5)
    
    # Convert acceleration into orientation + thrust
    # Transform pos/vel from NED frame to body frame 
    acc_des_b = apply_rot( rotz(-att[2,:]), acc_des )
    
    # Compute desired roll/pitch from desired acceleration
    T_des = LA.norm(acc_des_b, axis=0).reshape(1,-1) * mass
    # T_des = -acc_des_b[2,:].reshape(1,-1) * mass
    
    if np.isnan(T_des).any():
        print(f'att: {att.T}')
        print(f'acc_des: {acc_des.T}')
        print(f'acc_des_b: {acc_des_b.T}')
    
    phi_des   = np.arcsin(acc_des_b[1,:] / T_des)
    theta_des = -np.arctan2(acc_des_b[0,:], -acc_des_b[2,:]).reshape((1,-1))
    psi_des   = np.zeros(phi_des.shape)
    
    # Form att desired vector
    att_des = np.concatenate([phi_des, theta_des, psi_des], axis=0)
    att_des = add_trailing_dim(att_des)
    
    if np.isnan(att_des).any():
        print(acc_des_b[1,:] / T_des)
        raise ValueError(f'att_des {att_des} is nan')
    
    # Attitude Subsystem
    att_err  = ref_att + att_des - att 
    rate_err = ref_rate - rate
    ang_acc_des  = K_att @ np.concatenate([att_err, rate_err])
    LMN_des = MOI * ang_acc_des
    
    control_u = np.concatenate([T_des, LMN_des])
    
    if np.isnan(control_u).any():
        print(ref_att.T, att_des.T, att.T)
        print(att_err.T, rate_err.T)
        raise ValueError('control_u is nan')
    
    return control_u


# +
# Generate trajectories
def ref_function(t):
    if type(t) is not np.ndarray:
        t = np.array(t).reshape(1,1)
    
    # Get length N
    if t.ndim == 0:
        N = 1
    else:
        N = t.shape[0]
    
    step = -1 * (t > 0).astype(int).reshape((1,-1))
    sin1 = 4 * np.sin(1*t).reshape((1,-1))
    sin2 = 4 * np.sin(2*t).reshape((1,-1))
    zero = np.zeros((1,N))
    nan  = np.nan * zero.copy()
    
    ref_pos  = np.concatenate( [sin1, sin2, step] )
    ref_vel  = np.concatenate( [nan, nan, nan] )
    ref_att  = np.concatenate( [nan, nan, zero] )
    ref_rate = np.concatenate( [nan, nan, nan] )
    
    return np.concatenate( [ref_pos, ref_vel, ref_att, ref_rate] )

# Run Simulation
Ts = 0.01
t_span = [0,20]
t_eval = np.arange(t_span[0], t_span[1], Ts)
y0 = np.zeros(12)
sol = solve_ivp(eqn_of_motion, t_span, y0, t_eval=t_eval,
                args=(state_feedback, ref_function))

# Unpack solution
t = sol.t
pos  = sol.y[0:3,:]
vel  = sol.y[3:6,:]
att  = sol.y[6:9,:]
rate = sol.y[9:12,:]

# Reference
ref  = ref_function(sol.t)
ref_pos  = ref[0:3,:]
ref_vel  = ref[3:6,:]
ref_att  = ref[6:9,:]
ref_rate = ref[9:12,:]

# Control input
control_u = state_feedback(ref, sol.y)
T   = control_u[0,:]
LMN = control_u[1:4,:]

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

# + jupyter={"source_hidden": true} tags=[]
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
    time.sleep(0.005)

r = eul2rotm(np.array([2.20541314e-01, -1.29234325e-30, -5.88810570e-32]))
r.T @ np.array([0,0,2])
np.rad2deg(2.2e-1)
