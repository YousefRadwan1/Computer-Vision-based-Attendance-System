import pyttsx3
import threading

class VoiceFeedback:
    def __init__(self, rate=150, volume=0.8):
        """
        Initialize the voice feedback system.
        
        Args:
            rate (int): Speech rate (words per minute)
            volume (float): Volume level (0.0 to 1.0)
        """
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        
        # Get available voices and set a preferred voice if available
        voices = self.engine.getProperty('voices')
        if voices:
            # Try to use a female voice if available (often index 1)
            if len(voices) > 1:
                self.engine.setProperty('voice', voices[1].id)
        
        self.lock = threading.Lock()
    
    def speak(self, text):
        """
        Speak the given text asynchronously.
        
        Args:
            text (str): Text to be spoken
        """
        # Run in a separate thread to avoid blocking
        threading.Thread(target=self._speak_async, args=(text,), daemon=True).start()
    
    def _speak_async(self, text):
        """
        Internal method to speak text with thread safety.
        
        Args:
            text (str): Text to be spoken
        """
        with self.lock:
            self.engine.say(text)
            self.engine.runAndWait()