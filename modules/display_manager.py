import cv2
import numpy as np
import config

class DisplayManager:
    def __init__(self):
        """Initialize the display manager."""
        # Define colors for different attendance states
        self.NEW_ATTENDANCE_COLOR = config.NEW_ATTENDANCE_COLOR  # Green (0, 255, 0)
        self.REGULAR_ATTENDANCE_COLOR = config.REGULAR_ATTENDANCE_COLOR  # Blue (255, 0, 0)
        self.ALREADY_MARKED_COLOR = (0, 0, 255)  # Red for already marked students
    
    def draw_face_box(self, frame, face_coords, identity, emotion, newly_logged=False, already_marked=False):
        """
        Draw face detection box and information.
        
        Args:
            frame: Image frame to draw on
            face_coords: (x, y, w, h) coordinates of the face
            identity: Name of the recognized person
            emotion: Detected emotion
            newly_logged: Whether this is a newly logged attendance
            already_marked: Whether this student is already marked and within cooldown
            
        Returns:
            frame: The modified frame with annotations
        """
        if face_coords is None:
            return frame
        
        x, y, w, h = face_coords
        
        # Select color based on attendance status
        if already_marked:
            color = self.ALREADY_MARKED_COLOR  # Red for already marked students
            status_icon = "✗"  # X mark for already logged
        elif newly_logged:
            color = self.NEW_ATTENDANCE_COLOR  # Green for newly logged
            status_icon = "✓"  # Check mark for success
        else:
            color = self.REGULAR_ATTENDANCE_COLOR  # Blue for regular detection
            status_icon = ""
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Create background for text for better readability
        text = f"{identity} ({emotion})"
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, 1)
        
        # Draw semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y-text_height-10), (x+text_width, y), color, -1)
        alpha = 0.7  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        
        # Display name and emotion
        cv2.putText(frame, text, (x, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.FONT_COLOR, 2)
        
        # Add status icon if applicable
        if status_icon:
            cv2.putText(frame, status_icon, (x+w+10, y+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # For already marked students, show cooldown message
            if already_marked:
                cooldown_text = "Already logged"
                cv2.putText(frame, cooldown_text, (x, y+h+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ALREADY_MARKED_COLOR, 2)
        
        return frame
    
    def add_status_bar(self, frame, students_count):
        """
        Add status bar showing number of students logged.
        
        Args:
            frame: Image frame to draw on
            students_count: Number of students logged
            
        Returns:
            frame: The modified frame with status bar
        """
        # Create a black bar at the top of the frame
        status_bar_height = 40
        status_bar = np.zeros((status_bar_height, frame.shape[1], 3), dtype=np.uint8)
        
        # Add text for student count
        cv2.putText(status_bar, f"Students logged: {students_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add current time
        import time
        current_time = time.strftime("%H:%M:%S")
        time_text = f"Time: {current_time}"
        (text_width, _), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(status_bar, time_text, 
                    (frame.shape[1] - text_width - 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Combine status bar with the frame
        return np.vstack((status_bar, frame))
    
    def add_help_text(self, frame):
        """
        Add help text to the frame.
        
        Args:
            frame: Image frame to draw on
            
        Returns:
            frame: The modified frame with help text
        """
        help_text = [
            "Press 'q' to quit",
            "Press 'c' to calibrate camera",
            "Press 'a' to add new face"
        ]
        
        # Add help text at the bottom of the frame
        y_offset = frame.shape[0] - 10 - (len(help_text) * 30)
        for i, text in enumerate(help_text):
            cv2.putText(frame, text, (10, y_offset + i * 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def display_emotion_bars(self, frame, emotions_data):
        """
        Add emotion distribution bars to the frame.
        
        Args:
            frame: Image frame to draw on
            emotions_data: Dictionary of emotions and their counts
            
        Returns:
            frame: The modified frame with emotion bars
        """
        if not emotions_data:
            return frame
        
        # Calculate total for percentages
        total = sum(emotions_data.values())
        if total == 0:
            return frame
        
        # Define bar dimensions
        bar_height = 20
        max_bar_width = 150
        x_start = frame.shape[1] - max_bar_width - 10
        y_start = 50
        
        # Colors for emotions (BGR format)
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
        
        # Add title
        cv2.putText(frame, "Emotion Distribution:", 
                    (x_start, y_start - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw bars for each emotion
        for i, (emotion, count) in enumerate(emotions_data.items()):
            y_pos = y_start + i * (bar_height + 5)
            percentage = count / total
            bar_width = int(percentage * max_bar_width)
            
            # Draw bar
            color = emotion_colors.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (x_start, y_pos), (x_start + bar_width, y_pos + bar_height), 
                        color, -1)
            
            # Add text
            text = f"{emotion}: {int(percentage*100)}%"
            cv2.putText(frame, text, (x_start - 100, y_pos + bar_height - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def display_attendance_legend(self, frame):
        """
        Add a legend explaining the attendance status colors.
        
        Args:
            frame: Image frame to draw on
            
        Returns:
            frame: The modified frame with legend
        """
        # Position for the legend
        x_start = 10
        y_start = 50
        box_size = 15
        
        # Draw colored boxes with text
        # New attendance (green)
        cv2.rectangle(frame, (x_start, y_start), 
                     (x_start + box_size, y_start + box_size), 
                     self.NEW_ATTENDANCE_COLOR, -1)
        cv2.putText(frame, "Newly logged", (x_start + box_size + 5, y_start + box_size), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Already marked (red)
        cv2.rectangle(frame, (x_start, y_start + 25), 
                     (x_start + box_size, y_start + 25 + box_size), 
                     self.ALREADY_MARKED_COLOR, -1)
        cv2.putText(frame, "Already logged", (x_start + box_size + 5, y_start + 25 + box_size), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Regular detection (blue)
        cv2.rectangle(frame, (x_start, y_start + 50), 
                     (x_start + box_size, y_start + 50 + box_size), 
                     self.REGULAR_ATTENDANCE_COLOR, -1)
        cv2.putText(frame, "Detected", (x_start + box_size + 5, y_start + 50 + box_size), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame