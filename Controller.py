import numpy as np
import numpy.linalg as LA
from Utils import *

class StateFeedback:
    """
    Nested state feedback controllers
    1. Position + Velocity  --> 2. Attitude (Orientation) + Angular Rate

    Coordinate Frames
    - Position and Velocity are in NED
    """
    def __init__(self, kTranslation, KRotation):
        self.kTranslation = kTranslation
        self.KRotation = KRotation

    def __call__(self, ref, x):
        ref = ref.copy()
    
        # Physical parameters
        mass = 1
        MOI = np.diag([0.01, 0.01, 0.02])
        
        # ====================  Prepare Reference & State Inputs  ====================
        # Unpack state
        pos = x['pos']; vel  = x['vel']
        att = x['att']; rate = x['rate']
        eRb = x['eRb']; eul  = x['eul']
        
        # Unpack reference
        ref_pos  = ref[0:3]; ref_vel  = ref[3:6]
        ref_att  = ref[6:9]; ref_rate = ref[9:12]
        
        # Disable control loop if reference command all outer-loop commands are nan
        disable_pos = np.isnan(ref_pos)
        disable_vel = np.isnan(ref_vel) & disable_pos
        disable_att = np.isnan(ref_att) & disable_vel
        
        ref_pos[disable_pos] = pos[disable_pos]
        ref_vel[disable_vel] = vel[disable_vel]
        ref_att[disable_att] = eul[disable_att]
        
        # Replace the remaining NaNs with zero
        ref_vel[np.isnan(ref_vel) & ~disable_vel] = 0
        ref_att[np.isnan(ref_att) & ~disable_att] = 0
        ref_rate[np.isnan(ref_rate)] = 0
        
        # =============================  Translation  =============================
        # Position + Velocity Subsystem
        pos_err = ref_pos - pos
        vel_err = ref_vel - vel
        acc_des = self.kTranslation @ np.concatenate([pos_err, vel_err])
        
        # Add gravity
        acc_des[2] -= 9.81
        
        # Saturate acc desired
        acc_des[2] = np.clip(acc_des[2], -20, -5)
        
        # Transform acc_des to body frame
        acc_des_b = apply( rotz(-att[2]), acc_des )
        
        # =====================  Acceleration to Orientation  =====================
        def acc_to_att(acc_des_b):
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

            # Form att desired vector
            att_des = stack_squeeze([phi_des, theta_des, psi_des])
            
            return np.atleast_1d(T_des), att_des
            
        def acc_to_rotm(az):
            # Form desired rotation matrix
            edz = -az/LA.norm(az)
            # Form the ground projected (zero roll & pitch) x-axis
            x_vec = np.array([1.,0.,0.])
            vx = rotz(-att[2]) @ x_vec
            # Cross ez and vx to get the y-axis for the desired 1) acc_z and 2) heading
            edy = np.cross(edz, vx)
            # Finish the matrix by ey x ez
            edx = np.cross(edy, edz)
            
            # Stack (3*,N) vectors into (N,3*,3) rotations
            R_des = np.stack([ edx, edy, edz ], axis=1)
            
            # Let T_des be the dot(-acc_des, ez)
            ez = -np.squeeze(eRb[:,2])
            
            # Let desired thrust be <acc_des, ez>
            T_des = np.dot(acc_des_b, ez) * mass
            
            return np.atleast_1d(T_des), R_des

        
        # =============================  Rotation  =============================
            
        # Attitude Subsystem
        def attitude_state_feedback():
            att_err  = ref_att + att_des - eul 
            rate_err = ref_rate - rate

            ang_acc_des  = self.kRotation @ np.concatenate([att_err, rate_err])
            LMN_des = MOI @ ang_acc_des + np.cross(rate, MOI@rate) 

            return LMN_des
        
        def se3_geometric_control():
            # [PLACEHOLDER] Gains
            kR = 1000
            kOmega = 300
            
            eR = 0.5 * veemap(R_des.T @ eRb - eRb.T @ R_des)
            eOmega = rate - eRb.T @ R_des @ ref_rate
            
            LMN_des = MOI @ (-kR*eR - kOmega*eOmega) \
                    + np.cross(rate, MOI@rate) 
            
            return LMN_des
        
        T_des, att_des = acc_to_att(acc_des_b)
        LMN_des = attitude_state_feedback()
        
        # T_des, R_des = acc_to_rotm(acc_des_b)
        # if disable_vel[0:2].all():
        #     R_des = eul2rotm(ref_att)
        # LMN_des = se3_geometric_control()
        
        control_u = np.concatenate([T_des, LMN_des])
        return control_u