import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import pickle
import config
import mediapipe as mp
import time

class FaceRecognizer:
    def __init__(self):
        """Initialize the face detector module."""
        self.threshold = config.FACE_RECOGNITION_THRESHOLD
        self.face_data = {}
        self.face_data_file = os.path.join(config.KNOWN_FACES_DIR, "face_data.pkl")
        
        # Create known_faces directory if it doesn't exist
        if not os.path.exists(config.KNOWN_FACES_DIR):
            os.makedirs(config.KNOWN_FACES_DIR)
            print(f"Created directory: {config.KNOWN_FACES_DIR}")
        
        # Print the known_faces directory path for debugging
        abs_path = os.path.abspath(config.KNOWN_FACES_DIR)
        print(f"Using known faces directory: {abs_path}")
        
        # Load pre-existing face data if available
        self.load_face_data()
        
        # If no face data loaded, try to load from directory structure
        if not self.face_data:
            self.face_data = self.load_face_images()
            if self.face_data:
                print(f"Loaded face data from directory structure: {len(self.face_data)} persons")
                # Save the face data to pickle file for faster loading next time
                self.save_face_data()
        
        # Clean up missing images
        self.clean_missing_images()
        
        # Print summary of loaded faces
        print(f"Loaded {len(self.face_data)} persons' data")
        for name in self.face_data.keys():
            if "image_paths" in self.face_data[name]:
                paths = self.face_data[name]["image_paths"]
                print(f"  - {name}: {len(paths)} images")
                # Verify images exist
                valid_images = [p for p in paths if os.path.exists(p)]
                if len(valid_images) != len(paths):
                    print(f"    WARNING: Only {len(valid_images)} of {len(paths)} images exist!")
                
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
        
        # Cache detected faces to improve performance
        self.last_detection_time = 0
        self.last_detection_result = (None, None, 0)
        self.cache_timeout = 0.5  # seconds
        
        # Load haar cascade as backup
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def clean_missing_images(self):
        """Remove missing image files from face_data"""
        for name in list(self.face_data.keys()):
            if "image_paths" in self.face_data[name]:
                valid_paths = []
                for path in self.face_data[name]["image_paths"]:
                    if os.path.exists(path):
                        valid_paths.append(path)
                    else:
                        print(f"Removing missing image: {path}")
                
                # Update the image_paths list
                self.face_data[name]["image_paths"] = valid_paths
                
                # If no valid images remain, remove the person
                if not valid_paths:
                    print(f"Removing person with no valid images: {name}")
                    del self.face_data[name]
        
        # Save the cleaned data
        self.save_face_data()
        print("Cleaned face data saved")
    
    def load_face_data(self):
        """Load face data from pickle file if it exists."""
        if os.path.exists(self.face_data_file):
            try:
                with open(self.face_data_file, 'rb') as f:
                    self.face_data = pickle.load(f)
                print(f"Loaded {len(self.face_data)} face data entries")
            except Exception as e:
                print(f"Error loading face data: {e}")
                self.face_data = {}
    
    def load_face_images(self):
        """
        Load face images from the structured directory.
        Directory structure: known_faces_dir/student_name/image1.jpg, image2.jpg, etc.
        """
        if not os.path.exists(config.KNOWN_FACES_DIR):
            return {}
        
        face_data = {}
        
        # Get all subdirectories (each represents a person)
        try:
            subdirs = [d for d in os.listdir(config.KNOWN_FACES_DIR) 
                      if os.path.isdir(os.path.join(config.KNOWN_FACES_DIR, d)) and 
                      d != "__pycache__"]  # Skip __pycache__ directory
            
            for person_dir in subdirs:
                person_path = os.path.join(config.KNOWN_FACES_DIR, person_dir)
                person_name = person_dir  # Use directory name as person name
                
                # Valid image extensions
                valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                
                # Get all images for this person
                images = []
                for file in os.listdir(person_path):
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        image_path = os.path.join(person_path, file)
                        images.append(image_path)
                
                if images:
                    face_data[person_name] = {"image_paths": images, "timestamp": time.time()}
                    print(f"Loaded {len(images)} images for {person_name}")
            
            return face_data
        except Exception as e:
            print(f"Error loading face images from directories: {e}")
            return {}
    
    def save_face_data(self):
        """Save face data to pickle file."""
        try:
            with open(self.face_data_file, 'wb') as f:
                pickle.dump(self.face_data, f)
            print(f"Saved {len(self.face_data)} face data entries")
        except Exception as e:
            print(f"Error saving face data: {e}")
    
    def get_known_faces(self):
        """Return a list of names of known faces."""
        # Get names from face_data
        names_from_data = list(self.face_data.keys())
        
        # Also check for directory structure if face_data is empty
        if not names_from_data:
            # Get all directories in known_faces_dir
            if os.path.exists(config.KNOWN_FACES_DIR):
                try:
                    subdirs = [d for d in os.listdir(config.KNOWN_FACES_DIR) 
                              if os.path.isdir(os.path.join(config.KNOWN_FACES_DIR, d)) and
                              d != "__pycache__"]
                    return subdirs
                except Exception as e:
                    print(f"Error getting directories: {e}")
                    return []
        
        return names_from_data
    
    def detect_face(self, frame):
        """
        Detect faces using MediaPipe or OpenCV.
        
        Args:
            frame: The image frame to process
            
        Returns:
            tuple: face_coordinates or None if no face found
        """
        # First try MediaPipe
        face_coords = self.detect_face_mediapipe(frame)
        
        # If MediaPipe fails, try Haar Cascade
        if face_coords is None:
            face_coords = self.detect_face_haar(frame)
            
        return face_coords
    
    def detect_face_mediapipe(self, frame):
        """Detect faces using MediaPipe."""
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                # Get the first face detected
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to pixel coordinates
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                # Ensure coordinates are within frame boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)
                
                return (x, y, w, h)
            return None
        except Exception as e:
            print(f"MediaPipe face detection error: {e}")
            return None
    
    def detect_face_haar(self, frame):
        """Detect faces using Haar Cascade."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                return tuple(faces[0])  # Return the first face
            return None
        except Exception as e:
            print(f"Haar cascade face detection error: {e}")
            return None
    
    def recognize_face(self, frame):
        """
        Detect and recognize faces in the given frame.
        Uses enhanced template matching to identify faces from the known faces directory.
        
        Args:
            frame: The image frame to process
            
        Returns:
            tuple: (identity, face_coordinates, confidence)
        """
        # Check if we have a recent cached result
        current_time = time.time()
        if current_time - self.last_detection_time < self.cache_timeout:
            return self.last_detection_result
        
        try:
            face_coords = self.detect_face(frame)
            
            if face_coords is None:
                print("No face detected in frame")
                self.last_detection_time = current_time
                self.last_detection_result = (None, None, 0)
                return self.last_detection_result
            
            # If we have no known faces, return Unknown
            if not self.face_data:
                print("No face data available for recognition")
                self.last_detection_time = current_time
                self.last_detection_result = ("Unknown", face_coords, 0.5)
                return self.last_detection_result
            
            # Print known faces for debugging
            print(f"Available known faces: {list(self.face_data.keys())}")
            
            # Extract the detected face
            x, y, w, h = face_coords
            detected_face = frame[y:y+h, x:x+w]
            
            # Verify the face image is valid
            if detected_face.size == 0 or detected_face.shape[0] == 0 or detected_face.shape[1] == 0:
                print("Detected face region is invalid")
                self.last_detection_time = current_time
                self.last_detection_result = ("Unknown", face_coords, 0.5)
                return self.last_detection_result
            
            # Resize face to standard size for comparison
            detected_face_resized = cv2.resize(detected_face, (100, 100))
            detected_face_gray = cv2.cvtColor(detected_face_resized, cv2.COLOR_BGR2GRAY)
            
            # Variables to track best match
            best_match = None
            best_score = 0
            
            print("----- Starting face comparison -----")
            # Compare with each known face
            for name, person_data in self.face_data.items():
                if "image_paths" in person_data:
                    print(f"Comparing with {name} who has {len(person_data['image_paths'])} images")
                    person_scores = []
                    valid_image_count = 0
                    
                    # Compare with each image of this person
                    for img_path in person_data["image_paths"]:
                        try:
                            # Check if file exists
                            if not os.path.exists(img_path):
                                print(f"Image file not found: {img_path}")
                                continue
                                
                            # Load the known face image
                            known_face = cv2.imread(img_path)
                            if known_face is None:
                                print(f"Could not read image: {img_path}")
                                continue
                            
                            valid_image_count += 1
                                
                            # Resize and convert to grayscale
                            known_face_resized = cv2.resize(known_face, (100, 100))
                            known_face_gray = cv2.cvtColor(known_face_resized, cv2.COLOR_BGR2GRAY)
                            
                            # Try multiple template matching methods and take the best score
                            methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
                            method_scores = []

                            for method in methods:
                                result = cv2.matchTemplate(detected_face_gray, known_face_gray, method)
                                method_scores.append(np.max(result))

                            # Use the best score from any method
                            score = max(method_scores)
                            person_scores.append(score)
                            
                            print(f"  - Score for image {os.path.basename(img_path)}: {score:.4f}")
                            
                        except Exception as e:
                            print(f"Error comparing with image {img_path}: {e}")
                            continue
                    
                    # Get the average score for this person
                    if person_scores:
                        # Take the top 3 scores (or fewer if less than 3 images)
                        top_scores = sorted(person_scores, reverse=True)[:min(3, len(person_scores))]
                        avg_score = sum(top_scores) / len(top_scores)
                        print(f"Average score for {name}: {avg_score:.4f} from {valid_image_count} valid images")
                        
                        # Update best match if this person has a better score
                        if avg_score > best_score:
                            best_score = avg_score
                            best_match = name
                    else:
                        print(f"No valid comparisons for {name}")
            
            print(f"Best match: {best_match} with score {best_score:.4f}, threshold: {self.threshold}")
            
            # If the best score is above threshold, return the match
            if best_score > self.threshold:
                print(f"Recognized as {best_match} with confidence {best_score:.4f}")
                self.last_detection_time = current_time
                self.last_detection_result = (best_match, face_coords, best_score)
                return self.last_detection_result
            else:
                print(f"Unknown face, best match was {best_match} with score {best_score:.4f} (below threshold {self.threshold})")
                self.last_detection_time = current_time
                self.last_detection_result = ("Unknown", face_coords, best_score)
                return self.last_detection_result
                
        except Exception as e:
            print(f"Face recognition error: {e}")
            import traceback
            traceback.print_exc()
            self.last_detection_time = current_time
            self.last_detection_result = (None, None, 0)
            return self.last_detection_result
    
    def add_face(self, name, face_image):
        """
        Add a new face to the database with structured directory.
        
        Args:
            name: Name of the person
            face_image: Image containing their face
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create person directory if it doesn't exist
            person_dir = os.path.join(config.KNOWN_FACES_DIR, name)
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
                print(f"Created directory for {name}")
            
            # Generate a unique filename based on timestamp
            timestamp = int(time.time())
            filename = f"{timestamp}.jpg"
            file_path = os.path.join(person_dir, filename)
            
            # Save the face image
            cv2.imwrite(file_path, face_image)
            print(f"Saved face image to {file_path}")
            
            # Update face data
            if name in self.face_data:
                if "image_paths" not in self.face_data[name]:
                    self.face_data[name]["image_paths"] = []
                self.face_data[name]["image_paths"].append(file_path)
                self.face_data[name]["timestamp"] = time.time()
            else:
                self.face_data[name] = {
                    "image_paths": [file_path],
                    "timestamp": time.time()
                }
            
            # Save the updated data
            self.save_face_data()
            
            print(f"Added face for {name}")
            return True
        except Exception as e:
            print(f"Error adding face: {e}")
            return False
    
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()