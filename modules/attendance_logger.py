import os
import pandas as pd
from datetime import datetime
import config

# Import your existing voice feedback module instead of defining it here
from modules.voice_feedback import VoiceFeedback

class AttendanceLogger:
    def __init__(self):
        """Initialize the attendance logger module."""
        # Create attendance directory if it doesn't exist
        if not os.path.exists(config.ATTENDANCE_DIR):
            os.makedirs(config.ATTENDANCE_DIR)
            print(f"Created directory: {config.ATTENDANCE_DIR}")
        
        # Today's attendance file
        self.date_str = datetime.now().strftime("%Y-%m-%d")
        self.attendance_file = f"{config.ATTENDANCE_DIR}/attendance_{self.date_str}.csv"
        
        # Session attendance log (to prevent duplicates)
        self.attendance_log = {}
        
        # Initialize the voice feedback system
        self.voice_feedback = VoiceFeedback()
        
        # Initialize attendance file if it doesn't exist
        if not os.path.exists(self.attendance_file):
            df = pd.DataFrame(columns=["Name", "Time", "Emotion"])
            df.to_csv(self.attendance_file, index=False)
        else:
            # Load existing attendance to prevent duplicates across restarts
            try:
                existing_df = pd.read_csv(self.attendance_file)
                for _, row in existing_df.iterrows():
                    name = row["Name"]
                    time_str = row["Time"]
                    emotion = row["Emotion"]
                    # Convert time_str to datetime object
                    time_parts = time_str.split(":")
                    hour, minute, second = map(int, time_parts)
                    now = datetime.now()
                    timestamp = datetime(now.year, now.month, now.day, hour, minute, second)
                    
                    # Add to session log
                    self.attendance_log[name] = {
                        "timestamp": timestamp,
                        "emotion": emotion
                    }
                print(f"Loaded {len(self.attendance_log)} existing attendance records for today")
            except Exception as e:
                print(f"Error loading existing attendance: {e}")
    
    def check_already_logged(self, name):
        """
        Check if a student is already logged within the cooldown period.
        
        Args:
            name: Student's name
            
        Returns:
            tuple: (already_logged, minutes_remaining)
        """
        if name is None or name == "Unknown":
            return False, 0
            
        if name in self.attendance_log:
            current_time = datetime.now()
            last_time = self.attendance_log[name]["timestamp"]
            seconds_since_last_log = (current_time - last_time).total_seconds()
            
            if seconds_since_last_log < config.ATTENDANCE_COOLDOWN:
                minutes_remaining = int((config.ATTENDANCE_COOLDOWN - seconds_since_last_log) / 60)
                return True, minutes_remaining
                
        return False, 0
    
    def log_attendance(self, name, emotion):
        """
        Log a student's attendance.
        
        Args:
            name: Student's name
            emotion: Detected emotion
            
        Returns:
            bool: True if newly logged, False if already logged recently
        """
        # Skip if null name or "Unknown"
        if name is None or name == "Unknown":
            return False
        
        current_time = datetime.now()
        time_str = current_time.strftime("%H:%M:%S")
        
        # Check if already logged in current session
        already_logged, minutes_remaining = self.check_already_logged(name)
        if already_logged:
            message = f"{name} already logged. Can log again in {minutes_remaining} minutes."
            print(message)
            
            # Provide voice feedback for already logged student
            self.voice_feedback.speak(message)
            return False
        else:
            if name in self.attendance_log:
                print(f"Cooldown period elapsed for {name}. Logging attendance again.")
            else:
                print(f"First attendance log for {name} today.")
        
        # Add to session log
        self.attendance_log[name] = {
            "timestamp": current_time,
            "emotion": emotion
        }
        
        # Append to CSV file
        df = pd.DataFrame({
            "Name": [name],
            "Time": [time_str],
            "Emotion": [emotion]
        })
        
        df.to_csv(self.attendance_file, mode='a', header=False, index=False)
        
        # Log to console
        print(f"Logged attendance for {name} at {time_str} with emotion: {emotion}")
        
        # Provide voice feedback for successful attendance
        voice_message = f"{name} marked present, status: {emotion}"
        self.voice_feedback.speak(voice_message)
        
        return True
    
    def get_attendance_summary(self):
        """
        Get a summary of today's attendance.
        
        Returns:
            DataFrame: Attendance summary or None if no attendance file
        """
        if not os.path.exists(self.attendance_file):
            return None
        
        try:
            df = pd.read_csv(self.attendance_file)
            return df
        except Exception as e:
            print(f"Error reading attendance file: {e}")
            return None
        
    def export_attendance_report(self, output_file=None):
        """
        Export attendance report to CSV or Excel format.
        
        Args:
            output_file: File path for the report (default: generates based on date)
            
        Returns:
            str: Path to the exported file
        """
        summary = self.get_attendance_summary()
        if summary is None:
            return None
            
        if output_file is None:
            output_file = f"{config.ATTENDANCE_DIR}/report_{self.date_str}.xlsx"
            
        try:
            # Create a more detailed report
            report = summary.copy()
            
            # Add extra columns for analysis
            report['Hour'] = report['Time'].apply(lambda x: x.split(':')[0])
            
            # Group by hour to see attendance patterns
            hourly_stats = report.groupby('Hour').size().reset_index(name='Count')
            
            # Calculate emotion statistics
            emotion_stats = report.groupby('Emotion').size().reset_index(name='Count')
            
            # Create an Excel writer
            writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
            
            # Write each dataframe to a different worksheet
            report.to_excel(writer, sheet_name='Attendance', index=False)
            hourly_stats.to_excel(writer, sheet_name='Hourly Stats', index=False)
            emotion_stats.to_excel(writer, sheet_name='Emotion Stats', index=False)
            
            writer.close()
            print(f"Attendance report exported to {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error exporting attendance report: {e}")
            
            # Fallback to CSV if Excel export fails
            try:
                csv_file = f"{config.ATTENDANCE_DIR}/report_{self.date_str}.csv"
                summary.to_csv(csv_file, index=False)
                return csv_file
            except:
                return None