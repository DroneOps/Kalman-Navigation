"""
Kalman Filter implementation for position estimation. 

Author: DroneOps
Repository: Kalman-Navigation
"""

# Import necessary libraries
import numpy as np

# Kalman Filter class for position estimation
class KalmanPosicion:

    # Initialization method for the KalmanPosicion class
    def __init__(self, dt, std_acc, std_vision):
        """
        dt: Time step (in seconds)
        std_acc: Process noise (how much the velocity changes)
        std_vision: Measurement noise (how much we trust the ArUco detection)
        """
        self.dt = dt
        
        # Initial state vector [position, velocity] for the Kalman Filter
        self.x = np.array([[0.0], 
                           [0.0]])
        
        # State transition matrix (F) - Modified to not double the effect of velocity
        # The original F matrix would have been [[1, dt], [0, 1]], but since we are using velocity directly 
        # in the control input, we set the second row to zero to avoid double counting.
        self.F = np.array([[1, 0.0],
                           [0, 0.0]])
        
        # Control input matrix (B) - Adjusted for velocity control
        # Since the Tello provides velocity in cm/s, the change in position is: velocity * dt
        # The sign is NEGATIVE because moving forward (+vel) reduces the distance to the marker.
        self.B = np.array([[-self.dt],
                           [1.0]])
        
        # Measurement matrix (H) - We only measure position, not velocity
        self.H = np.array([[1, 0]])
        
        # Error covariance matrix (P) - Initialized with high uncertainty
        self.P = np.eye(2) * 500
        
        # Process noise covariance matrix (Q) - Represents the uncertainty in the process model
        self.Q = np.array([[self.dt**2, 0.0],
                           [0.0, 1.0]]) * std_acc**2
        
        # Measurement noise covariance matrix (R) - Represents the uncertainty in the measurement model
        self.R = std_vision

    # Prediction step of the Kalman Filter
    def predict(self, velocity):

        # x = F*x + B*u
        # When passing a positive 'velocity', B will handle subtracting distance in the state x[0][0]
        self.x = np.dot(self.F, self.x) + np.dot(self.B, velocity)
        
        # P = F*P*F' + Q
        # The covariance update uses a modified F matrix to reflect the actual dynamics of the system, 
        # ensuring that the uncertainty in position is updated correctly based on the process noise.
        F_dynamic = np.array([[1, -self.dt], [0, 1]]) # Dynamic F matrix for covariance update
        self.P = np.dot(np.dot(F_dynamic, self.P), F_dynamic.T) + self.Q
        
        return self.x[0][0] # Returns the predicted position (x[0][0]) after the prediction step

    # Update step of the Kalman Filter
    def update(self, pos_measured):

        # S = H*P*H' + R
        # The innovation covariance S combines the uncertainty in the predicted state (P) and the measurement noise (R) 
        # to determine how much we should trust the new measurement
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # K = P*H' * inv(S) (Kalman Gain)
        # The Kalman Gain K determines how much we should adjust our state estimate based on the new measurement. 
        # A higher K means we trust the measurement more, while a lower K means we trust our prediction more.
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # y = measurement - prediction (innovation)
        # The innovation y represents the difference between the actual measurement and the predicted measurement.
        y = pos_measured - np.dot(self.H, self.x)
        
        # Update state and covariance
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

        # Return the updated position estimate after incorporating the new measurement
        return self.x[0][0]