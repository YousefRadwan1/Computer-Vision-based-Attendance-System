import os
import cv2
import numpy as np
import glob
import config

class CameraCalibrator:
    def __init__(self):
        """Initialize the camera calibration module."""
        # Create calibration directory if it doesn't exist
        if not os.path.exists(config.CALIBRATION_DIR):
            os.makedirs(config.CALIBRATION_DIR)
            print(f"Created directory: {config.CALIBRATION_DIR}")
        
        self.camera_matrix = None
        self.dist_coeffs = None
        self.load_calibration()
    
    def load_calibration(self):
        """Load camera calibration data if available."""
        if os.path.exists(config.CALIBRATION_FILE):
            try:
                data = np.load(config.CALIBRATION_FILE)
                self.camera_matrix = data["camera_matrix"]
                self.dist_coeffs = data["dist_coeffs"]
                print("Camera calibration loaded successfully.")
                return True
            except Exception as e:
                print(f"Error loading calibration data: {e}")
        return False
    
    def find_chessboard_enhanced(self, gray, CHECKERBOARD):
        """
        Enhanced chessboard detection using multiple methods.
        
        Args:
            gray: Grayscale image
            CHECKERBOARD: Chessboard size (width, height)
            
        Returns:
            success: Boolean indicating if chessboard was found
            corners: Detected corners if found, None otherwise
        """
        # Method 1: Standard detection
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret:
            return True, corners
        
        # Method 2: Try with adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        ret, corners = cv2.findChessboardCorners(thresh, CHECKERBOARD, None)
        if ret:
            return True, corners
        
        # Method 3: Try with blurred image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, corners = cv2.findChessboardCorners(blurred, CHECKERBOARD, None)
        if ret:
            return True, corners
        
        return False, None
    
    def create_chessboard_image(self, output_path, square_size=80, width=7, height=6):
        """
        Create a chessboard calibration pattern image.
        
        Args:
            output_path: Path to save the generated chessboard image
            square_size: Size of each square in pixels
            width: Number of inner corners horizontally (7 for an 8×7 board)
            height: Number of inner corners vertically (6 for an 8×7 board)
        """
        # Board dimensions in squares (add 1 to inner corners)
        board_width = width + 1
        board_height = height + 1
        
        # Create image with correct dimensions
        img = np.zeros(((board_height) * square_size, 
                        (board_width) * square_size), dtype=np.uint8)
        
        # Fill alternating squares with white
        for i in range(board_height):
            for j in range(board_width):
                if (i + j) % 2 == 0:
                    y1, y2 = i * square_size, (i + 1) * square_size
                    x1, x2 = j * square_size, (j + 1) * square_size
                    img[y1:y2, x1:x2] = 255
        
        cv2.imwrite(output_path, img)
        print(f"Chessboard pattern saved as '{output_path}'")
        return img

    def calibrate_camera(self, cap):
        """
        Calibrate the camera using a chessboard pattern.
        
        Args:
            cap: OpenCV VideoCapture object
            
        Returns:
            bool: True if calibration successful, False otherwise
        """
        print("Starting camera calibration...")
        
        # Defining the dimensions of chessboard
        CHECKERBOARD = config.CHECKERBOARD_SIZE
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = []
        
        # Defining the world coordinates for 3D points
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        
        if not cap.isOpened():
            print("Cannot open camera")
            return False
        
        # Generate a sample chessboard image if needed
        sample_path = os.path.join(config.CALIBRATION_DIR, "chessboard_sample.png")
        if not os.path.exists(sample_path):
            self.create_chessboard_image(sample_path, width=CHECKERBOARD[0], height=CHECKERBOARD[1])
            print(f"A sample chessboard pattern has been generated at {sample_path}")
            print("Please print this pattern or display it on another screen for calibration.")
        
        img_counter = 0
        needed_images = config.CALIBRATION_FRAMES_NEEDED
        
        print(f"Please show a {CHECKERBOARD[0]}x{CHECKERBOARD[1]} chessboard pattern.")
        print(f"Will take {needed_images} images. Press 'c' to capture when chessboard is visible.")
        print("Press 'q' to quit, 'r' to reset counter.")
        
        while img_counter < needed_images:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Try to find chessboard corners for preview
            preview_frame = frame.copy()
            found, corners = self.find_chessboard_enhanced(gray, CHECKERBOARD)
            
            # Display calibration instructions and feedback
            cv2.putText(preview_frame, f"Captured: {img_counter}/{needed_images}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if found:
                cv2.putText(preview_frame, "CHESSBOARD DETECTED! Press 'c' to capture", 
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Draw the corners for preview
                preview_frame = cv2.drawChessboardCorners(preview_frame, CHECKERBOARD, corners, found)
            else:
                cv2.putText(preview_frame, "No chessboard detected. Move the pattern or adjust lighting.", 
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.putText(preview_frame, "Press 'q' to quit, 'r' to reset counter", 
                      (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Calibration', preview_frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                img_counter = 0
                objpoints = []
                imgpoints = []
                print("Reset counter and points arrays.")
            elif key == ord('c'):
                # If already found in preview, use those corners
                if found:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)
                    
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(frame.copy(), CHECKERBOARD, corners2, found)
                    
                    # Save the captured image for documentation
                    capture_path = os.path.join(config.CALIBRATION_DIR, f"calib_img_{img_counter}.jpg")
                    cv2.imwrite(capture_path, frame)
                    
                    cv2.imshow('Captured Chessboard', img)
                    cv2.waitKey(500)
                    
                    img_counter += 1
                    print(f"Image {img_counter} captured and saved to {capture_path}")
                else:
                    print("Chessboard not found. Try again.")
        
        cv2.destroyAllWindows()
        
        if img_counter > 0:
            print("Calibrating camera...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)
            
            # Save calibration data
            np.savez(config.CALIBRATION_FILE, camera_matrix=mtx, dist_coeffs=dist)
            self.camera_matrix = mtx
            self.dist_coeffs = dist
            
            # Calculate reprojection error
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                mean_error += error
            
            print(f"Camera calibration completed with average reprojection error: {mean_error/len(objpoints)}")
            return True
        else:
            print("Calibration failed - insufficient images captured.")
            return False
    
    def apply_calibration(self, frame):
        """
        Apply camera calibration to undistort an image.
        
        Args:
            frame: Input image frame
            
        Returns:
            frame: Calibrated frame or original if no calibration data
        """
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
            
            # Undistort
            dst = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
            
            # Crop the image
            x, y, w, h = roi
            if all(v > 0 for v in [x, y, w, h]):
                dst = dst[y:y+h, x:x+w]
                # Resize back to original size if needed
                dst = cv2.resize(dst, (frame.shape[1], frame.shape[0]))
            return dst
        return frame