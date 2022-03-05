import numpy as np
import numpy.linalg as LA
from flightcontrol.Utils import *

class Controller:
    """
    Abstract class for flight controllers

    Provides disable_control_loops method, which disables loops based on NaNs in the reference input.

    Must override this class. 
    """
    def __init__(self, params=None):
        self.params = params if params else { 'mass':1, 'MOI':np.array([0.01, 0.01, 0.02]) }

    @staticmethod
    def disable_control_loops(ref, x):
        """
        Enable/disable control loops based on NaNs in the reference command

        Disable control loop if all outer-loop commands are nan, otherwise set to zero
        """
        # Find commands to disable
        disable_pos = np.isnan(ref['pos'])
        disable_vel = np.isnan(ref['vel']) & disable_pos
        disable_att = np.isnan(ref['att']) & disable_vel
        
        # Disconnect control loop by setting ref = state 
        ref['pos'][disable_pos] = x['pos'][disable_pos]
        ref['vel'][disable_vel] = x['vel'][disable_vel]
        ref['att'][disable_att] = x['eul'][disable_att]
        
        # Replace the remaining with zero
        ref['vel'][np.isnan(ref['vel']) & ~disable_vel] = 0
        ref['att'][np.isnan(ref['att']) & ~disable_att] = 0
        ref['rate'][np.isnan(ref['rate'])] = 0

        return ref

    def __call__(self, ref, x):
        raise NotImplementedError('Subclasses must override Controller!')

class PIDController(Controller):
    """
    Nested state feedback controllers
    1. Position + Velocity  --> 2. Attitude (Orientation) + Angular Rate

    Coordinate frames of position and velocity are in NED
    """
    def __init__(self, kTranslation, kRotation, params=None):
        super().__init__(params)
        self.kTranslation = kTranslation
        self.kRotation = kRotation

    def acc_to_att(self, acc_des_b):
        # Compute desired roll/pitch from desired acceleration
        T_des = np.atleast_1d(LA.norm(acc_des_b, axis=0)) * self.params['mass']
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
        att_des = np.squeeze(np.stack([phi_des, theta_des, psi_des]))
        
        return np.atleast_1d(T_des), att_des

    def __call__(self, ref, x):
        # ====================  Prepare Reference & State Inputs  ====================
        ref = self.disable_control_loops(ref.copy(), x)
        
        # =============================  Translation  =============================
        # Position + Velocity Subsystem
        pos_err = ref['pos'] - x['pos']
        vel_err = ref['vel'] - x['vel']
        acc_des = self.kTranslation @ np.concatenate([pos_err, vel_err])
        
        # Add gravity
        acc_des[2] -= 9.81
        
        # Saturate acc desired
        acc_des[2] = np.clip(acc_des[2], -20, -5)
        
        # Transform acc_des to body frame
        acc_des_b = apply( rotz(-x['att'][2]), acc_des )
        
        # =====================  Acceleration to Orientation  =====================
        T_des, att_des = self.acc_to_att(acc_des_b)
        
        # =============================  Rotation  =============================
        # Attitude Subsystem
        att_err  = ref['att'] - x['eul'] + att_des
        rate_err = ref['rate'] - x['rate']

        ang_acc_des  = self.kRotation @ np.concatenate([att_err, rate_err])
        LMN_des = self.params['MOI'] * ang_acc_des \
                + np.cross(x['rate'], self.params['MOI'] * x['rate'])
        
        return np.concatenate([T_des, LMN_des])