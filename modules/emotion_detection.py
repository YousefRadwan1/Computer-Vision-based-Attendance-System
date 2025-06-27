import cv2
import numpy as np
from deepface import DeepFace

class EmotionDetector:
    def __init__(self):
        """Initialize the emotion detector module."""
        # List of emotions we want to track (based on DeepFace's output)
        self.target_emotions = ['happy', 'sad', 'neutral', 'angry', 'surprise', 'fear', 'disgust']
        
        # Preload models to avoid loading delay during first detection
        try:
            print("Initializing emotion detection model...")
            # Preload the model with a dummy detection
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False, silent=True)
            print("Emotion detection model loaded")
        except Exception as e:
            print(f"Warning: Could not preload emotion model: {e}")
    
    def detect_emotion(self, frame, face_coords):
        """
        Detect emotion from a face in the given coordinates.
        
        Args:
            frame: The image frame
            face_coords: Tuple of (x, y, w, h) face coordinates
            
        Returns:
            str: Detected emotion or "unknown" if detection failed
        """
        if face_coords is None:
            return "unknown"
        
        try:
            x, y, w, h = face_coords
            
            # Ensure coordinates are within frame bounds
            frame_h, frame_w = frame.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame_w - x)
            h = min(h, frame_h - y)
            
            # Make sure the face region is large enough for analysis
            if w < 48 or h < 48:
                return "unknown"
            
            # Extract face region with a margin for better detection
            margin = 20
            y_start = max(0, y - margin)
            y_end = min(frame_h, y + h + margin)
            x_start = max(0, x - margin)
            x_end = min(frame_w, x + w + margin)
            
            face_img = frame[y_start:y_end, x_start:x_end]
            
            if face_img.size == 0:  # Empty image
                return "unknown"
            
            # Use DeepFace for emotion analysis
            result = DeepFace.analyze(
                img_path=face_img,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            
            # DeepFace can return a list or dict depending on the version
            if isinstance(result, list):
                emotions = result[0]['emotion']
            else:
                emotions = result['emotion']
            
            # Find the emotion with the highest score
            predominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            # Print detected emotions for debugging
            print(f"Detected emotions: {emotions}")
            print(f"Predominant emotion: {predominant_emotion}")
            
            return predominant_emotion
        
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return "unknown"
    
    def get_emotion_color(self, emotion):
        """
        Get a color associated with an emotion for display purposes.
        
        Args:
            emotion: String representing the emotion
            
        Returns:
            tuple: BGR color values
        """
        # Color mapping (BGR format)
        emotion_colors = {
            'happy': (0, 255, 255),    # Yellow
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'neutral': (255, 255, 255),# White
            'surprise': (0, 255, 0),   # Green
            'fear': (255, 0, 255),     # Purple
            'disgust': (0, 128, 128),  # Brown
            'unknown': (128, 128, 128) # Gray
        }
        
        return emotion_colors.get(emotion, (255, 255, 255))