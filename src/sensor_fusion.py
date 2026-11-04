"""
Script for sensor fusion using Kalman filter with IMU and ArUco vision data.

Author: DroneOps
Repository: Kalman-Navigation
"""


# Import necessary libraries
import cv2
import cv2.aruco as aruco
import numpy as np
import time
from djitellopy import Tello
from kalman import KalmanPosicion    # Import Kalman filter class
from vision_z_y import ArucoDetector # Import ArUco detector class

def main():

    # Initialize Tello drone
    tello = Tello()

    # Attempt to connect and start video stream
    try:
        tello.connect()         # Connect to the drone
        tello.streamon()        # Start video stream
        print(f"Established connection. Battery: {tello.get_battery()}%")

    # Handle connection errors
    except Exception as e:
        print(f"Connection error: {e}")
        return

    # Parameters 
    #   * dt: 0.05 (20Hz aprox) 
    #   * std_acc: ruido IMU 
    #   * std_vision: ruido ArUco
    # Initialize Kalman filter for position estimation
    kf = KalmanPosicion(dt=0.05, std_acc=2.0, std_vision=10.0)
    
    detector = ArucoDetector(aruco.DICT_6X6_50)         # Initialize ArUco detector with 6x6 dictionary
    frame_read = tello.get_frame_read()                 # Get the video frame reader from Tello
    
    last_time = time.time()                             # Initialize last_time for dt calculation

    # Print initialization message
    print("Initializing Sensor Fusion... Press 'q' to quit.")

    # Main loop for sensor fusion
    try:

        # Loop indefinitely until user interrupts
        while True:

            # Calculate time difference for Kalman filter update
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            kf.dt = dt # Update the dt dynamically for greater precision

    # --- Obtain IMU data from Tello (Prediction step) ---
            # The acceleration in X is what pushes the drone towards the window
            acc_x = tello.get_acceleration_x() 
            
            # Predict the next position using the Kalman filter based on IMU data
            pos_predicted = kf.predict(np.array([[acc_x]]))

    # --- Obtain vision data from ArUco markers (Correction step) ---
            frame = frame_read.frame    # Get the current video frame from Tello

            # If a frame is available, process it for ArUco detection and sensor fusion
            if frame is not None:

                # Convert the frame to BGR and grayscale for ArUco detection
                img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                # Detect ArUco markers in the grayscale image
                corners, ids, _ = detector.detector.detectMarkers(gray)

                # Initialize variable to hold the visual distance measurement
                visual_dist = None

                # If at least 4 markers are detected, proceed with PnP and Kalman update
                if ids is not None and len(ids) >= 4:

                    img_points = []                     # Initialize list to hold image points for PnP
                    ids_f = ids.flatten()               # Flatten the detected IDs for easier processing (converting a multi-dimensional array into a one-dimensional array)
                    target_ids = sorted(ids_f)[:4]      # Select the first 4 detected IDs for processing

                    # For each target ID, find the corresponding corner points and compute the mean position
                    for tid in target_ids:
                        idx = np.where(ids_f == tid)[0][0]              # Find the index of the target ID
                        img_points.append(corners[idx][0].mean(axis=0)) # Compute the mean of the corner points for the target ID
                    
                    # Convert the list of image points to a NumPy array of type float32 for PnP processing
                    img_points = np.array(img_points, dtype=np.float32)

                    # Solve the PnP problem to estimate the pose of the detected markers
                    success, rvec, tvec = cv2.solvePnP(
                        detector.obj_points,            # 3D object points of the ArUco markers
                        img_points,                     # 2D image points of the detected markers
                        detector.camera_matrix,         # Camera matrix
                        detector.dist_coeffs,           # Distortion coefficients
                        flags=cv2.SOLVEPNP_ITERATIVE    # Use the iterative method for solving PnP
                    )

                    # If PnP was successful, extract the Z distance from the translation vector and update the Kalman filter
                    if success:

                        # Extract the Z distance from the translation vector (tvec) which represents the distance to the detected ArUco markers
                        visual_dist = tvec[2][0]
                        
                        # Update the Kalman filter with the visual distance measurement from PnP
                        fused_pos = kf.update(visual_dist)
                        color_msg = (0, 255, 0)         # Green: IMU + Vision (PnP successful)

                    # If PnP failed, use the predicted position from the Kalman filter without correction
                    else:
                        fused_pos = pos_predicted
                        color_msg = (0, 165, 255)       # Orange: IMU + Vision (PnP failed, using prediction)

                # If less than 4 markers are detected, use the predicted position from the Kalman filter without correction
                else:

                    fused_pos = pos_predicted
                    color_msg = (0, 0, 255)             # Red: IMU only (no vision data available)

                # --- Visualization ---
                output = detector.process_frame(img_bgr) # Process the frame to visualize detected markers and errors

                # Display the fused position from the Kalman filter on the output frame
                cv2.putText(output, f"Kalman X: {fused_pos:.1f} cm", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_msg, 2)

                # Display the status of the sensor fusion (IMU + Vision or IMU only) on the output frame
                status = "VISION+IMU" if visual_dist else "IMU ONLY(PREDICTION)"

                # Display the status message on the output frame
                cv2.putText(output, status, (10, 95), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_msg, 1)

                # Display the output frame with all visualizations in a window titled "Sensor Fusion Window Detection"
                cv2.imshow("Sensor Fusion Window Detection", output)

            # If q is pressed, exit the loop and terminate the program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Handle user interruption (Ctrl+C)
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    
    # Handle any other exceptions that may occur during execution
    finally:
        tello.streamoff()
        cv2.destroyAllWindows()
        tello.end()

if __name__ == "__main__":
    main()