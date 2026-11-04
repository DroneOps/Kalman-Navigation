"""
This script is responsible for detecting ArUco markers, estimating their 3D pose, and calculating the error in 
pixels and distance in centimeters. It uses OpenCV's ArUco module for marker detection and pose estimation. 
The script also includes a main function for offline testing with a standard webcam.

Author: DroneOps
Repository: Kalman-Navigation
"""

# Import essential dependencies
import cv2
import cv2.aruco as aruco
import numpy as np

# Class definition for ArUco detection and pose estimation
class ArucoDetector:

    # Initialization method that sets up the ArUco dictionary, detector parameters, and camera calibration
    def __init__(self, aruco_type, frame=None):
        
        self.aruco_dict = aruco.getPredefinedDictionary(aruco_type)         # The dictionary is used for detecting specific ArUco markers
        self.parameters = aruco.DetectorParameters()                        # Set the parameters for the ArUco detector
        self.error = (0, 0, 0)                                              # Initialize the error tuple
        self.frame = frame

        # Set the default frame size for processing (can be adjusted based on camera resolution)
        self.x = 400
        self.y = 300
        
        # Set specific parameters for the ArUco detector to improve detection accuracy
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX         # Uses subpixel refinement for corner detection
        self.parameters.cornerRefinementWinSize = 5                                 # Sets the window size for corner refinement
        self.parameters.adaptiveThreshConstant = 2                                  # Sets the constant for adaptive thresholding to improve marker detection under varying lighting conditions
        self.parameters.minMarkerPerimeterRate = 0.01                               # Sets the minimum perimeter rate for detected markers to filter out small false positives

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)   # Initializes the ArUco detector with the specified dictionary and parameters

        # Camera calibration parameters (intrinsic matrix and distortion coefficients) are scaled based on the frame size
        scale_x = self.x / 960.0
        scale_y = self.y / 720.0
        ajuste_focal = 0.8

        # Obtain the camera matrix coefficients (focal lengths and principal point)
        self.fx = 917.2963798156287 * scale_x * ajuste_focal
        self.fy = 918.5664777675206 * scale_y * ajuste_focal
        self.cx = 489.60023141280055 * scale_x
        self.cy = 373.24114282133826 * scale_y

        # Set up the camera matrix and distortion coefficients for pose estimation
        self.camera_matrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array([[-0.012, 0.115, 0.001, -0.001, -0.32]], dtype=np.float32)

        # Define the real-world width of the ArUco window in centimeters for accurate pose estimation
        self.real_width_cm = 38.0 
        hs = self.real_width_cm / 2                # Half of the real width to define the 3D coordinates of the marker corners in the object space
       
        # Define the 3D coordinates of the marker corners in the object space (assuming the marker is on the XY plane at Z=0)
        self.obj_points = np.array([[-hs, hs, 0], [hs, hs, 0], [hs, -hs, 0], [-hs, -hs, 0]], dtype=np.float32)

    # Method to detect ArUco markers in the input frame, estimate their pose, and calculate errors in pixels and distance
    def detect_markers(self, frame):
        
        # Resize the input frame to the specified dimensions and convert it to grayscale for marker detection
        self.frame = cv2.resize(frame, (self.x, self.y), interpolation=cv2.INTER_AREA)      # Resize the frame
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)                                 # Convert the frame to grayscale for marker detection

        # Detect ArUco markers in the grayscale image and obtain their corners and IDs
        corners, ids, rejected = self.detector.detectMarkers(gray)    # Detects ArUco markers in the grayscale image
        center_window = (self.x // 2, self.y // 2)                    # Calculate the center of the window for reference in error calculations 
        
        # Draw crosshairs in the center of the frame for visual reference
        cv2.line(self.frame, (self.x // 2, 0), (self.x // 2, self.y), (255, 0, 255), 1)
        cv2.line(self.frame, (0, self.y // 2), (self.x, self.y // 2), (255, 0, 255), 1)
        cv2.circle(self.frame, center_window, 4, (255, 255, 255), -1)

        # Draw individual ArUco markers in green if detected
        if ids is not None:
            for i in range(len(ids)):
                c = corners[i][0]
                cv2.putText(self.frame, f'ID: {ids[i][0]}', (int(c[0][0]), int(c[0][1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.polylines(self.frame, [corners[i].astype(int)], True, (0, 255, 0), 1)

        # If at least 4 markers are detected, proceed with pose estimation algorithm
        if ids is not None and len(ids) >= 4:

            # Flatten the IDs array and select the first 4 unique IDs for pose estimation
            ids_f = ids.flatten()                       # Flattening means to convert the 2D array of IDs into a 1D array for easier processing
            target_ids = sorted(ids_f)[:4]              # Select the first 4 unique IDs for pose estimation (sorted to maintain consistency in order)
            img_points = []                             # Initialize an empty list to store the corresponding image points for the detected markers

            # For each target ID, find the corresponding corner in the detected markers and calculate the mean position of the corners to get a single point for each marker
            for tid in target_ids:
                idx = np.where(ids_f == tid)[0][0]
                img_points.append(corners[idx][0].mean(axis=0))

            # Convert the list of image points to a NumPy array of type float32 for compatibility with OpenCV functions    
            img_points = np.array(img_points, dtype=np.float32)

            # Use the solvePnP function to estimate the pose of the ArUco markers based on the object points and image points
            success, rvec, tvec = cv2.solvePnP(
                self.obj_points,                  # 3D coordinates of the marker corners in the object space
                img_points,                       # 2D coordinates of the detected marker corners in the image space
                self.camera_matrix,               # Camera intrinsic matrix containing focal lengths and principal point
                self.dist_coeffs,                 # Distortion coefficients for the camera lens
                flags=cv2.SOLVEPNP_ITERATIVE      # Use the iterative method for pose estimation, which is suitable for small numbers of points and provides good accuracy
            )

            # If the pose estimation is successful, calculate the distance and error metrics for navigation
            if success:

                # Calculate the frontal distance (Z-axis) from the translation vector obtained from solvePnP
                dist_z = tvec[2][0]                                     # Real frontal distance in cm (Drone's X-axis)
                center_px = np.mean(img_points, axis=0).astype(int)     # Calculate the center of the detected markers in pixel coordinates (mean of the image points)

                # Calculate the pixel error in the Y-axis (horizontal) based on the center of the detected markers and the center of the window
                error_pixel_y = center_px[0] - center_window[0]
                
                # Calculate the pixel offset in the Z-axis (vertical) based on the real frontal distance and the camera's focal length
                offset_pixels_z = int((7.5 * self.fy) / max(1.0, dist_z))

                # Calculate the adjusted center of the window in the Z-axis by subtracting the pixel offset from the center of the window
                center_window_ajustado_z = center_window[1] - offset_pixels_z  
                
                # Calculate the pixel error in the Z-axis (vertical) based on the adjusted center of the window and the center of the detected markers
                error_pixel_z = center_window_ajustado_z - center_px[1]

                # Calculate the distance error in the X-axis (frontal) by converting the real frontal distance to an integer value
                error_dist_x = int(dist_z)

                # Store the calculated errors in a tuple for later retrieval (Kalman algorithm)
                self.error = (error_pixel_y, error_pixel_z, error_dist_x)

                # Draw the detected markers and their IDs on the frame for visual feedback
                cv2.polylines(self.frame, [img_points.astype(int)], True, (255, 100, 0), 2)
                
                # Draw a red circle at the center of the detected markers for visual reference
                cv2.circle(self.frame, tuple(center_px), 5, (0, 0, 255), -1)
                
                # Draw a yellow line from the adjusted center of the window to the center of the detected markers to visualize the error in the Z-axis
                cv2.line(self.frame, (center_window[0], center_window_ajustado_z), tuple(center_px), (0, 255, 255), 2)

                # Display the calculated distance and error metrics on the frame for real-time feedback
                cv2.putText(self.frame, f"Real X: {dist_z:.1f} cm", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.putText(self.frame, f"Err Y: {self.error[0]} px", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                cv2.putText(self.frame, f"Err Z (Ajustado -6cm): {self.error[1]} px", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        else:

            # If fewer than 4 markers are detected, reset the error to zero and display a message indicating that the system is searching for markers
            self.error = (0, 0, 0)
            cv2.putText(self.frame, "BUSCANDO MARCO...", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return self.frame

    def get_error(self):
        return self.error 
    
def main():

    # Create an instance of the ArucoDetector class with the specified ArUco dictionary type for testing using a webcam
    detector = ArucoDetector(aruco.DICT_6X6_50) 
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the captured frame to detect ArUco markers and estimate their pose, then display the processed frame in a window
        processed_frame = detector.detect_markers(frame)
        cv2.imshow('Prueba Local Coordinates', processed_frame)

        # Check for the 'q' key press to exit the loop and close the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()