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

def eul2rotm(eul):
    """Rotation matrix from euler angles"""
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

def apply_rot(rot_matrix, vec):
    """
    Apply a sequence of rotation matrices (Nx3x3)
    to either:
    - a vector (3,)
    - a sequence of vector (3,N)
    """
    # Check for the single Rot matrix case (3,3) and return instantly
    if rot_matrix.ndim < 3:
        return rot_matrix @ vec
    
    # Apply N rotations to a single vector
    if vec.ndim == 1:
        for R in rot_matrix:
            return R @ vec
        
    # Apply N rotations to N vectors
    elif rot_matrix.shape[0] == vec.shape[1]:
        N = rot_matrix.shape[0]
        output = np.zeros((3,N))
        for i,R in enumerate(rot_matrix):
            output[:,i] = R @ vec[:,i]
        return output
    
    # N rotations cannot be applied to M vectors
    else:
        raise ValueError("apply_rot : Different number of rotation matrices and vectors received")

def atan2(Y, X) -> np.ndarray:
    """Numpy's arctan2, but output is at least 1D array"""
    return np.atleast_1d(np.arctan2(Y,X))

def stack_squeeze(arr):
    """Stack along axis 0, then squeeze to remove any trailing dimensions of size 1"""
    return np.squeeze(np.stack( arr ))


# -

# **Setup**
# 1. Multicopter physics
# 2. Generate trajectories
# 3. Define control law

# + tags=[]
# Define Multicopter Physics
def eqn_of_motion(t, y, control_law, ref_function):
    mass = 1
    J_inv = np.array([1/0.01, 1/0.01, 1/0.02])
    T_max = 30
    LMN_max = 1
    
    # Unpack state
    pos = y[0:3]
    vel = y[3:6]
    att = y[6:9]
    rate = y[9:12]
    
    eRb = eul2rotm(att)
    
    # Call trajectory and control functions
    ref = ref_function(t)
    control_u = control_law(ref, y)
    
    # ================  Actuator Model  =================
    T = np.atleast_1d(control_u[0])
    LMN = control_u[1:4]
    
    T = np.clip(T, 0, T_max)
    LMN = np.clip(LMN, -LMN_max, LMN_max)
    
    #[DEBUG] print(f'T: {T.shape}')
    
    thrust = np.concatenate([0*T, 0*T, -T])
    
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
    
    #[DEBUG] print(f'd_pos: {d_pos.shape} \t d_vel: {d_vel.shape}')
    #[DEBUG] print(f'd_att: {d_att.shape} \t d_rate: {d_rate.shape}')
    
    dydt = np.concatenate([d_pos, d_vel, d_att, d_rate], axis=0)
    
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
    pos  = x[0:3]
    vel  = x[3:6]
    att  = x[6:9]
    rate = x[9:12]
    
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
    acc_des_b = apply_rot( rotz(-att[2]), acc_des )
    
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
sol = solve_ivp(eqn_of_motion, t_span, y0, t_eval=t_eval,
                args=(state_feedback, ref_function))

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


