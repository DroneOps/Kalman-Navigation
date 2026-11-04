"""
This module is responsible for managing the drone's operations, including takeoff, landing, and sending 
control signals based on the Kalman Filter's output.

Author: DroneOps
Repository: Kalman-Navigation
"""

# Import necessary libraries
import Control_Aruco
import threading                  # Allows for concurrent execution of tasks
import time
from djitellopy import Tello

# Kalman Filter class for position estimation
class MissionControl:

    # Initialization method for the MissionControl class, which sets up the Tello drone, ArUco controller, and threading for speed monitoring.
    def __init__(self):
        self.tello = Tello()      # Initialize the Tello drone object
        self.tello.connect()      # Connect to the Tello drone
        self.tello.streamon()     # Start the video stream from the Tello drone
        self.cap = self.tello.get_frame_read()      # Get the video frame reader from the Tello drone
        
        # Initialize the ArUco controller with the MissionControl instance and the video frame reader
        self.aruco_controller = Control_Aruco.ArucoController(self, self.cap)
        
        self.lock = threading.Lock()                # Lock for thread-safe operations
        self.running = True                         # Flag to control the running state of the mission
        
    # Method to retrieve the current speeds of the drone in the x, y, and z directions
    def get_speeds(self):
        speed_x = self.tello.get_speed_x()
        speed_y = self.tello.get_speed_y()
        speed_z = self.tello.get_speed_z()
        return speed_x, speed_y, speed_z
    
    # Method to command the drone to take off
    def takeoff(self):
        self.tello.takeoff()
    
    # Method to command the drone to land
    def land(self):
        self.tello.land()
    
    # Method to save the drone's speeds continuously
    def save_speeds(self):

        # Continue running this loop as long as the mission is active
        while self.running:

            # Try to get the speeds and sleep for a short duration to avoid overwhelming the drone's API
            try: 
                _ = self.tello.get_speed_x()
                _ = self.tello.get_speed_y()
                _ = self.tello.get_speed_z()
                time.sleep(0.1)

            # If an error occurs, sleep for a longer duration before retrying
            except: 
                time.sleep(0.5)
    
    # Method to start the ArUco program
    def start_mission(self):

        # Start the ArUco controller's main loop
        try:
            self.aruco_controller.run() 

        # If a keyboard interrupt is detected, stop the mission and land the drone
        except KeyboardInterrupt:      
            pass

        # Ensure that the drone is safely stopped and landed, that the video stream is turned off and the connection is ended
        finally:

            # Stop the drone's movement and land it safely
            try:
                self.tello.send_rc_control(0, 0, 0, 0) # Stop all movement
                self.land()                            # Land the drone safely

            # If an error occurs during landing, just pass and ensure the drone is stopped
            except: 
                pass

            # Stop the video stream and end the connection with the drone
            try: 
                self.tello.streamoff()                 # Stop the video stream
                self.tello.end()                       # End the connection with the drone

            # If an error occurs while stopping the video stream or ending the connection, just pass
            except: 
                pass

    # Method to send control signals to the drone based on the calculated control signals for y, z, and x directions
    def send_control_signals(self, control_signal_y, control_signal_z, control_signal_x):
        
        # Send the calculated control signals to the drone using the Tello SDK's send_rc_control method
        """
        control_signal_y: Control signal for the y-axis (left/right movement)
        control_signal_z: Control signal for the z-axis (up/down movement)
        control_signal_x: Control signal for the x-axis (forward/backward movement)
        """
        self.tello.send_rc_control(int(control_signal_y), int(control_signal_x), int(control_signal_z), 0)

# Main execution block to start the mission control and speed monitoring threads
if __name__ == "__main__":
    mission_control = MissionControl() # Create an instance of the MissionControl class
    
    t1 = threading.Thread(target=mission_control.start_mission) # Create a thread for the ArUco mission control
    t2 = threading.Thread(target=mission_control.save_speeds)   # Create a thread for continuously saving the drone's speeds
    
    t1.daemon = True # Automatically close when the main program exits
    t2.daemon = True # Automatically close when the main program exits

    # Start both threads and keep the main program running while the mission is active
    try:
        
        t1.start() # Start the ArUco mission control thread
        t2.start() # Start the speed monitoring thread

        # Keep the main program running while the mission is active, checking if the ArUco mission control thread is still alive
        while t1.is_alive():
            time.sleep(0.5)

    # If a keyboard interrupt is detected, stop the mission and land the drone
    except KeyboardInterrupt:

        # Set the running flag to False to stop the speed monitoring thread
        mission_control.running = False

        # Stop the drone's movement and land it safely
        try:
            mission_control.tello.send_rc_control(0, 0, 0, 0)
            mission_control.land()

        # If an error occurs during landing, just pass and ensure the drone is stopped
        except:
            pass
    
    # Ensure that the drone is safely stopped and landed, that the video stream is turned off and the connection is ended    
    finally:

        # End Tello video stream and connection
        try:
            mission_control.tello.streamoff()
            mission_control.tello.end()

        # If an error occurs while stopping the video stream or ending the connection, just pass
        except:
            pass
