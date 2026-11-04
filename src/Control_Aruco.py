"""
This script is the core autonomous navigation loop for flying through an ArUco window frame. The algorithm integrates
a PI controller as well as a Kalman Filter coupled with Sensor Fusion. 

Author: DroneOps
Repository: Kalman-Navigation
"""

# Import essential dependencies
import cv2
import numpy as np

# Import custom classes and scripts
import coordinatesAruco             # Performs ArUco detection and Point-n-Perspective (PnP) estimation to compute spatial error
import MissionControl as MC         # Handles takeoff, sending RC velocity commands and reading speeds from drone IMU
from kalman import KalmanPosicion   # Kalman filter class that predicts and updates position/velocities states per axis

# Class initialization for top-module controller
class ArucoController:

    # Initialization method recieves mission_control object and frame handler
    def __init__(self, mission_control=None, frame=None):

        # Definition of crucial objects 
        self.aruco = coordinatesAruco.ArucoDetector(cv2.aruco.DICT_6X6_50, frame)  # Initialization of Aruco object
        self.cap = frame                                                           # Definition of frame handler
        self.mission_control = mission_control                                     # Definition of mission control object

        # Definition of global variables for the PI controller
        self.integral_y = 0          # Stores the integral or the sum of the error for Y axis
        self.integral_z = 0          # Stores the integral or the sum of the error for Z axis
        self.errorPrev = (0, 0, 0)   # Array that stores the previous error for all axes
        
        # Control gains for centering (Y and Z axes)
        self.Kpz = 0.50     # Proportional gain for Z axis
        self.Kiz = 0.001    # Steady offset gain for Z axis
        self.Kpy = 0.50     # Proportional gain for Y axis
        self.Kiy = 0.001    # Steady offset gain for Y axis

        # Base forward cruise speed
        self.base_forward_speed = 15 

        # Time step (main loops runs at 10 Hz or 100 ms time execution)
        self.dt = 0.1 

        # Initialization of independent Kalman Filters per axis
        # 'std_acc': standard deviation for process noise
        # 'std_vision': standard deviation for measurement noise
        self.kalman_y = KalmanPosicion(dt=self.dt, std_acc=0.2, std_vision=4.0) 
        self.kalman_z = KalmanPosicion(dt=self.dt, std_acc=0.2, std_vision=3.0) 
        self.kalman_x = KalmanPosicion(dt=self.dt, std_acc=0.1, std_vision=2.0) 
        
        # Safety flag to prevent lockups during takeoff
        self.frame_detected_at_least_once = False

    # Accepts error as parameter and outputs control signal after applying controller calculations
    def PI_control(self, error):

        # Proportional and integral errors are added
        self.proportional_y = round(self.Kpy * error[0], 3) # error[0] = horizontal displacement
        self.proportional_z = round(self.Kpz * error[1], 3) # error[1] = vertical displacement

        self.integral_y += round(self.Kiy * error[0] * self.dt, 3) 
        self.integral_z += round(self.Kiz * error[1] * self.dt, 3) 

        control_signal_y = int(self.proportional_y + self.integral_y)
        control_signal_z = int(self.proportional_z + self.integral_z)

        return control_signal_y, control_signal_z

    def run(self):
        # Takeoff method from mission control object
        self.mission_control.takeoff() 

        # Acts as a failsafe, if the videoframe drops for 30 frames, the main loop is broken
        frame_timeout = 0
        max_timeout = 30 
        
        # Velocities configurations
        self.base_forward_speed = 12  # Slow speed for aligning with the ArUco
        self.sprint_speed = 15  # Sprint speed for when the drone loses window from sight

        # Begin main loop
        while True:

            # Set frame unless no capture is deteced
            frame = self.cap.frame if self.cap is not None else None   # If no capture is detected, frame is set to None

            # Set frame timout if no capture is detected
            if frame is None:
                frame_timeout += 1 
                if frame_timeout > max_timeout: 
                    break # End loop if timeout is exceeded
                continue 

            # Reset frame timeout upon detection
            frame_timeout = 0 
            
            # Visual processing using PnP
            processed_frame = self.aruco.detect_markers(frame) # Return processed frame with telemetry and error calculations
            raw_error = self.aruco.get_error()                 # Returns the raw error in pixels and distance in cm (error_y, error_z, distance_x)

            # Obtain speeds directly from onboard IMU avoiding UDP deadlocks
            try:
                vel_y, vel_z, vel_x = self.mission_control.get_speeds()  # Obtain current velocities
            except:
                vel_y, vel_z, vel_x = 0, 0, 0                            # Set velocities to zero if speeds cannot be obtained

            # Sensor fusion and Kalman filtering for error estimation and control signal generation
            if raw_error != (0, 0, 0):

                # First case: drone has detected ArUco window
                self.frame_detected_at_least_once = True            # Mark that the marker has been detected at least once

                # Kalman prediction step using current velocities
                self.kalman_y.predict(vel_y)                            # Predict the next state for the Y axis using current velocity
                self.kalman_z.predict(vel_z)                            # Predict the next state for the Z axis ...
                self.kalman_x.predict(vel_x)                            # Predict the next state for the X axis ...

                # Kalman update step using raw error measurements
                filtered_err_y = self.kalman_y.update(raw_error[0])     # Update the current state for the Y axis using raw error measurement
                filtered_err_z = self.kalman_z.update(raw_error[1])     # Update the current state for the Z axis ...
                estimated_dist_x = self.kalman_x.update(raw_error[2])   # Update the current state for the X axis ...

                # Generate control signals using the PI controller based on filtered errors
                filtered_error = (filtered_err_y, filtered_err_z, estimated_dist_x)  # Create a tuple of filtered errors
                cs_y, cs_z = self.PI_control(filtered_error)            # Generate control signals for Y and Z axes using the PI controller
                
                # Set gentle forward speed to help with alignment
                cs_x = self.base_forward_speed
                
            else:
                # Second phase: drone loses ArUco window detection, but is set to cross the window before filter uncertainty accumulates
                if self.frame_detected_at_least_once:

                    # Freeze lateral and vertical signals so the drone stops swaying
                    self.kalman_y.x = np.array([[0.0], [0.0]])
                    self.kalman_z.x = np.array([[0.0], [0.0]])
                    
                    # Estimate remaining distance using only current velocity
                    estimated_dist_x = self.kalman_x.predict(vel_x)

                    # Control signals are set to zero 
                    cs_y = 0
                    cs_z = 0
                    
                    # Increase forward speed to cross window before filter uncertainty accumulates
                    cs_x = self.sprint_speed
                    
                    filtered_err_y = 0.0
                    filtered_err_z = 0.0
                else:
                    # Drone stays stationary if no window detected upon takeoff
                    estimated_dist_x = 999.0                # Set a high value to indicate that the drone is far from the window
                    cs_y = 0
                    cs_z = 0
                    cs_x = 0 
                    filtered_err_y = 0.0
                    filtered_err_z = 0.0

            # Halt motion as soon as the drone moves 25 cm behind the window
            if estimated_dist_x <= -25.0:
                self.mission_control.send_control_signals(0, 0, 0)  # Send zero control signals to stop the drone
                break

            # Live telemetry visuals
            if self.frame_detected_at_least_once:
                cv2.putText(processed_frame, f"Est. Kalman X: {estimated_dist_x:.1f} cm", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                if raw_error == (0, 0, 0):
                    cv2.putText(processed_frame, "¡CRUISING SPEED!", (250, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Send control signals
            self.mission_control.send_control_signals(cs_y, cs_z, cs_x)

            # Update previous error for next iteration
            self.errorPrev = (filtered_err_y, filtered_err_z, estimated_dist_x)

            # Display the processed frame with telemetry and error visuals
            cv2.imshow('Aruco Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()