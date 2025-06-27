import os
import cv2
import numpy as np
import time
from datetime import datetime
from collections import Counter

# Import modules
from modules.face_recognition import FaceRecognizer
from modules.emotion_detection import EmotionDetector
from modules.attendance_logger import AttendanceLogger
from modules.camera_calibration import CameraCalibrator
from modules.display_manager import DisplayManager
from modules.voice_feedback import VoiceFeedback
import config

class AttendanceSystem:
    def __init__(self):
        """Initialize the attendance system."""
        # Create required directories
        for directory in [config.KNOWN_FACES_DIR, config.ATTENDANCE_DIR, config.CALIBRATION_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
        
        # Initialize modules
        self.face_recognizer = FaceRecognizer()
        self.emotion_detector = EmotionDetector()
        self.attendance_logger = AttendanceLogger()
        self.camera_calibrator = CameraCalibrator()
        self.display_manager = DisplayManager()
        
        # State variables
        self.add_face_mode = False
        self.new_face_name = ""
        self.emotions_history = []  # Track emotions for statistics
        
        # Startup message
        welcome_message = "Attendance system initialized and ready"
        self.attendance_logger.voice_feedback.speak(welcome_message)
    
    def update_emotion_stats(self, emotion):
        """Update emotion statistics."""
        if emotion and emotion != "unknown":
            self.emotions_history.append(emotion)
            # Keep only the last 100 emotions for statistics
            if len(self.emotions_history) > 100:
                self.emotions_history = self.emotions_history[-100:]
    
    def get_emotion_distribution(self):
        """Get the distribution of emotions."""
        if not self.emotions_history:
            return {}
        return dict(Counter(self.emotions_history))
    
    def add_new_face(self, frame, face_coords):
        """Add a new face to the system."""
        if face_coords is None:
            print("No face detected for enrollment")
            self.attendance_logger.voice_feedback.speak("No face detected for enrollment")
            return False
        
        if not self.new_face_name:
            self.new_face_name = input("Enter name for the new face: ")
            if not self.new_face_name:
                print("Name cannot be empty")
                self.attendance_logger.voice_feedback.speak("Name cannot be empty")
                self.add_face_mode = False
                return False
        
        x, y, w, h = face_coords
        face_img = frame[y:y+h, x:x+w]
        
        result = self.face_recognizer.add_face(self.new_face_name, face_img)
        
        # Provide voice feedback on face addition
        if result:
            self.attendance_logger.voice_feedback.speak(f"Face for {self.new_face_name} added successfully")
        else:
            self.attendance_logger.voice_feedback.speak("Failed to add face")
            
        self.add_face_mode = False
        self.new_face_name = ""
        return result
    
    def run(self):
        """Run the attendance system."""
        # Initialize camera
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        
        if not cap.isOpened():
            print("Cannot open camera")
            self.attendance_logger.voice_feedback.speak("Cannot open camera")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        print("Attendance System Started")
        print("Press 'q' to quit, 'c' to calibrate camera, 'a' to add new face")
        self.attendance_logger.voice_feedback.speak("Attendance System Started")
        
        # Track last announcement time for already logged students to avoid spam
        last_announcement = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting...")
                self.attendance_logger.voice_feedback.speak("Camera disconnected. Exiting.")
                break
            
            # Apply camera calibration if available
            processed_frame = self.camera_calibrator.apply_calibration(frame)
            
            # If in add face mode, only detect face without recognition
            if self.add_face_mode:
                try:
                    # Use the same face detection that works in normal mode
                    # We'll just discard the identity but keep the face coordinates
                    _, face_coords, _ = self.face_recognizer.recognize_face(processed_frame)
                    
                    if face_coords:
                        x, y, w, h = face_coords
                        
                        # Draw rectangle around the face
                        cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(processed_frame, "Press 'c' to capture this face", 
                                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        key = cv2.waitKey(1)
                        if key == ord('c'):
                            if self.add_new_face(processed_frame, face_coords):
                                print("Face added successfully")
                            else:
                                print("Failed to add face")
                            self.add_face_mode = False
                    else:
                        cv2.putText(processed_frame, "No face detected. Position your face in the camera.", 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                except Exception as e:
                    print(f"Error in face detection: {e}")
                    self.attendance_logger.voice_feedback.speak("Error in face detection")
                    self.add_face_mode = False
            
            else:
                # Normal mode: Recognize face and detect emotion
                identity, face_coords, confidence = self.face_recognizer.recognize_face(processed_frame)
                
                # If face detected, detect emotion and possibly log attendance
                emotion = "unknown"
                newly_logged = False
                already_marked = False
                
                if face_coords:  # A face was detected
                    emotion = self.emotion_detector.detect_emotion(processed_frame, face_coords)
                    self.update_emotion_stats(emotion)
                    
                    if identity and identity != "Unknown":  # Only log attendance for known faces
                        # Check if already logged without trying to log yet
                        already_marked, minutes_remaining = self.attendance_logger.check_already_logged(identity)
                        
                        if already_marked:
                            # Use red box for already marked students
                            current_time = time.time()
                            
                            # Control voice announcements to avoid spam
                            # Only announce every 10 seconds for the same person
                            if (identity not in last_announcement or 
                                current_time - last_announcement.get(identity, 0) > 10):
                                self.attendance_logger.voice_feedback.speak(
                                    f"{identity} already logged. Can log again in {minutes_remaining} minutes.")
                                last_announcement[identity] = current_time
                            
                            # Draw with "already marked" status
                            processed_frame = self.display_manager.draw_face_box(
                                processed_frame, face_coords, identity, emotion, False, True)
                        else:
                            # Not in cooldown, proceed with normal logging
                            newly_logged = self.attendance_logger.log_attendance(identity, emotion)
                            
                            # Draw rectangle around face with details
                            processed_frame = self.display_manager.draw_face_box(
                                processed_frame, face_coords, identity, emotion, newly_logged, False)
                    else:
                        # Still display the face box for unknown faces, but don't log attendance
                        processed_frame = self.display_manager.draw_face_box(
                            processed_frame, face_coords, "Unknown", emotion, False, False)
            
            # Add status bar and help text
            students_count = len(self.attendance_logger.attendance_log)
            processed_frame = self.display_manager.add_status_bar(processed_frame, students_count)
            processed_frame = self.display_manager.add_help_text(processed_frame)
            
            # Add emotion distribution bars
            emotion_distribution = self.get_emotion_distribution()
            processed_frame = self.display_manager.display_emotion_bars(processed_frame, emotion_distribution)
            
            # Add legend for color coding
            processed_frame = self.display_manager.display_attendance_legend(processed_frame)
            
            # Show the frame
            cv2.imshow('Attendance System', processed_frame)
            
            # Process keyboard input
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.attendance_logger.voice_feedback.speak("Stopping attendance system")
                break
            elif key == ord('c') and not self.add_face_mode:
                self.attendance_logger.voice_feedback.speak("Starting camera calibration")
                self.camera_calibrator.calibrate_camera(cap)
            elif key == ord('a') and not self.add_face_mode:
                self.add_face_mode = True
                self.new_face_name = ""  # Reset name when entering add face mode
                print("Add face mode activated. Position face in the camera.")
                self.attendance_logger.voice_feedback.speak("Add face mode activated. Position face in the camera")
            elif key == ord('r'):
                # Export report feature
                report_path = self.attendance_logger.export_attendance_report()
                if report_path:
                    self.attendance_logger.voice_feedback.speak("Attendance report generated successfully")
                else:
                    self.attendance_logger.voice_feedback.speak("Failed to generate attendance report")
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Attendance System Stopped")

if __name__ == "__main__":
    # Initialize and run the system
    system = AttendanceSystem()
    
    # Check if there are any images in the known_faces directory
    known_faces = system.face_recognizer.get_known_faces()
    if not known_faces:
        print("Warning: No student images found in the database.")
        print("You can add student faces using the 'a' key during execution.")
        system.attendance_logger.voice_feedback.speak("Warning: No student images found in the database. You can add student faces using the A key.")
        choice = input("Continue anyway? (y/n): ")
        if choice.lower() != 'y':
            system.attendance_logger.voice_feedback.speak("Exiting attendance system")
            exit()
    
    system.run()