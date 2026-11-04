"""
This script reads the IMU data from the Tello drone, including attitude (pitch, roll, yaw), acceleration, and speed. 
It continuously prints this information to the console until interrupted by the user.

Author: DroneOps
Repository: Kalman-Navigation
"""

# Import essential dependencies
from djitellopy import Tello
import time

# Main function to read and display IMU data from the Tello drone
def main():
    
    tello = Tello()                     # Create an instance of the Tello class to interface with the drone


    print("Connecting to Tello...")     # Print a message indicating the connection process
    tello.connect()                     # Establish a connection with the Tello drone

    # Check the battery level and print it to the console to check connection status
    battery = tello.get_battery()
    print(f"Connection established. Battery level: {battery}%")

    # Print a message indicating that the IMU reading process is starting and how to stop it
    print("\nInitializing IMU reading... (Press Ctrl+C to stop)\n")

    # Start an infinite loop to continuously read and display IMU data
    try:
        while True:
            # Return the current attitude of the drone in degrees
            pitch = tello.get_pitch()              # Pitch: rotation around the X-axis
            roll = tello.get_roll()                # Roll: rotation around the Y-axis
            yaw = tello.get_yaw()                  # Yaw: rotation around the Z-axis

            # Return the current acceleration of the drone in centimeters per second squared (cm/s²)
            acc_x = tello.get_acceleration_x()
            acc_y = tello.get_acceleration_y()
            acc_z = tello.get_acceleration_z()

            # Return the current speed of the drone in centimeters per second (cm/s)
            vel_x = tello.get_speed_x()
            vel_y = tello.get_speed_y()
            vel_z = tello.get_speed_z()

            # Print the IMU data in a formatted manner to the console
            print(f"Attitude      -> Pitch: {pitch:4}° | Roll: {roll:4}° | Yaw: {yaw:4}°")
            print(f"Acceleration  -> X: {acc_x:6.1f} | Y: {acc_y:6.1f} | Z: {acc_z:6.1f} (cm/s²)")
            print(f"Velocity      -> X: {vel_x:6.1f} | Y: {vel_y:6.1f} | Z: {vel_z:6.1f} (cm/s)")
            print("-" * 65)

            # Sleep for 0.5 seconds before the next reading to avoid overwhelming the console with data
            time.sleep(0.5)

    # Handle exceptions to allow for graceful exit and error reporting
    except KeyboardInterrupt:
        # Handle the case where the user interrupts the program with Ctrl+C
        print("\nUser interrupted the program.")
        
    except Exception as e:
        # Handle any unexpected errors that may occur during execution
        print(f"\nAn unexpected error occurred: {e}")
        
    finally:
        # Ensure that the Tello drone is properly disconnected and resources are released
        print("Disconnecting...")
        tello.end()

if __name__ == "__main__":
    main()