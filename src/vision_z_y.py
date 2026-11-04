"""
Vision-based Z and Y Position Estimation for ArUco Markers.

Author: DroneOps
Repository: Kalman-Navigation
"""

# Import necessary libraries
import cv2
import cv2.aruco as aruco
import numpy as np
from djitellopy import Tello

# Define the ArUcoDetector class for detecting ArUco markers and estimating their position
class ArucoDetector:
    
    # Initialize the ArUcoDetector with a specific ArUco dictionary type
    def __init__(self, aruco_type):
        
        # Initialize the ArUco dictionary and detector parameters
        self.aruco_dict = aruco.getPredefinedDictionary(aruco_type)
        self.parameters = aruco.DetectorParameters()
        
        # Set the corner refinement method to subpixel accuracy for better detection
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5         # Set the window size for corner refinement to 5 pixels.
                                                            # Higher values may improve accuracy but can slow down detection.
        
        # Set the adaptive threshold constant to 2 for better marker detection under varying lighting conditions.
        self.parameters.adaptiveThreshConstant = 2          # This parameter helps in adjusting the thresholding for marker detection. 
        self.parameters.minMarkerPerimeterRate = 0.01       # Set the minimum marker perimeter rate to 0.01 to filter out very small markers that may be noise.

        # Initialize the ArUco detector
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        # Set the default frame size for processing (adjust based on camera resolution)
        self.x, self.y = 400, 300
        scale_x = self.x / 960.0
        scale_y = self.y / 720.0
        
        # Set the camera intrinsic parameters (focal lengths and principal point) based on the camera calibration.
        focal_adjustment = .8
        self.fx = 917.2963798156287 * scale_x * focal_adjustment
        self.fy = 918.5664777675206 * scale_y * focal_adjustment
        self.cx = 489.60023141280055 * scale_x
        self.cy = 373.24114282133826 * scale_y

        # Set the camera matrix and distortion coefficients for the camera used in the Tello drone.
        self.camera_matrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32) 
        self.dist_coeffs = np.array([[-0.012, 0.115, 0.001, -0.001, -0.32]], dtype=np.float32)

        # Define the real-world dimensions of the ArUco marker (in centimeters) for accurate distance estimation.
        self.real_width_cm = 38.0 
        hs = self.real_width_cm / 2
        
        # Define the 3D coordinates of the marker corners in the marker's local coordinate system.
        self.obj_points = np.array([[-hs, hs, 0], [hs, hs, 0], [hs, -hs, 0], [-hs, -hs, 0]], dtype=np.float32)

    # Process a video frame to detect ArUco markers and estimate their position.
    def process_frame(self, frame):
        
        # Resize the input frame to the specified dimensions for processing and convert it to grayscale for marker detection.
        self.frame = cv2.resize(frame, (self.x, self.y), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers in the grayscale image and obtain their corner coordinates and IDs.
        corners, ids, rejected = self.detector.detectMarkers(gray)#_filtered)

        # If at least 4 markers are detected, proceed with pose estimation using the PnP algorithm.
        if ids is not None and len(ids) >= 4:

            # Flatten the detected IDs for easier processing and select the first 4 detected IDs for pose estimation.
            ids_f = ids.flatten()
            target_ids = sorted(ids_f)[:4]

            # Initialize a list to hold the image points corresponding to the detected marker corners for PnP processing.
            img_points = []

            # For each target ID, find the corresponding corner points and compute the mean position of the corners
            for tid in target_ids:
                idx = np.where(ids_f == tid)[0][0]
                img_points.append(corners[idx][0].mean(axis=0))
            
            # Convert the list of image points to a NumPy array of type float32 for PnP processing.
            img_points = np.array(img_points, dtype=np.float32)

            # Solve the PnP problem to estimate the pose of the detected markers
            success, rvec, tvec = cv2.solvePnP(
                self.obj_points,                # 3D object points of the ArUco markers
                img_points,                     # 2D image points of the detected markers
                self.camera_matrix,             # Camera matrix
                self.dist_coeffs,               # Distortion coefficients
                flags=cv2.SOLVEPNP_ITERATIVE    # Use the iterative method for solving PnP
            )

            # If PnP was successful, extract the Z distance from the translation vector and visualize the results
            if success:
                
                # Extract the Z distance from the translation vector (tvec)
                dist_z = tvec[2][0] 
                
                # Calculate the center of the detected marker in pixel coordinates by averaging the image points
                center_px = np.mean(img_points, axis=0).astype(int)

                # Calculate the center of the window (frame) in pixel coordinates
                center_window = (self.x // 2, self.y // 2)

                # Calculate the error in pixels between the detected marker center and the window center
                error_x = center_px[0] - center_window[0]
                error_y = center_window[1] - center_px[1]  # Invert the Y-axis error to match the image coordinate system

                # Draw a rectangle as a background for the text to improve visibility
                cv2.rectangle(self.frame, (220, 5), (395, 60), (0, 0, 0), -1) # Background for text

                # Display the error in pixels for both X and Y axes on the output frame
                cv2.putText(self.frame, f"Err: X {error_x} px", (230, 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(self.frame, f"Err: Y {error_y} px", (230, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Draw a line that connects both centers
                cv2.line(self.frame, center_window, tuple(center_px), (0, 255, 255), 2)

                # Draw a circle at the center of the detected marker and draw the detected marker's outline using polylines
                cv2.circle(self.frame, tuple(center_px), 7, (0, 0, 255), -1)              # Center of the detected marker
                cv2.polylines(self.frame, [img_points.astype(int)], True, (0, 255, 0), 2) # Detected marker outline

                # Display the Z distance from the camera to the detected marker on the output frame
                cv2.putText(self.frame, f"Distancia: {dist_z:.1f} cm", (10, 30), 
                            1, 1.2, (0, 255, 0), 2)                                    
        
        # If less than 4 markers are detected, display a message indicating that the system is searching for markers
        else:
            cv2.putText(self.frame, "BUSCANDO...", (10, 30), 1, 1.0, (0, 0, 255), 2)

        # Return the processed frame with all visualizations
        return self.frame

def main():

    # A Tello object is created to interface with the Tello drone
    tello = Tello()

    # Attempt to connect to the Tello drone and start the video stream
    try:
        tello.connect()
        tello.streamon()

    # Any exceptions that occur during the connection or streaming process are caught and handled gracefully
    except:
        return

    # Create an instance of the ArucoDetector class with the specified ArUco dictionary type
    detector = ArucoDetector(aruco.DICT_6X6_50)
    
    # Continuously read frames from the Tello drone's camera and process them for ArUco marker detection and visualization
    frame_read = tello.get_frame_read()

    # The main loop runs indefinitely until the user interrupts it (e.g., by pressing 'q' or closing the window)
    while True:

        # Get the current frame from the Tello drone's camera
        frame = frame_read.frame

        # If no frame is available, skip the current iteration and continue to the next loop iteration
        if frame is None: continue

        # Convert the frame from RGB to BGR color space for OpenCV processing
        img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Process the frame using the ArucoDetector to detect markers and visualize the results
        output = detector.process_frame(img_bgr)
        
        # Display the processed output frame in a window titled "Final Test"
        cv2.imshow("Final Test", output)
        
        # Check if the 'q' key is pressed to exit the loop and terminate the program
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # After exiting the loop, turn off the video stream and close all OpenCV windows to clean up resources
    tello.streamoff()
    
    # Close all OpenCV windows to free up resources and ensure a clean exit from the program
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()