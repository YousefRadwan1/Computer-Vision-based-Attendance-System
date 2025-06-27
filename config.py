# Configuration file for the attendance system

# Paths
KNOWN_FACES_DIR = "data/known_faces"
ATTENDANCE_DIR = "data/attendance"
CALIBRATION_DIR = "data/calibration"
CALIBRATION_FILE = f"{CALIBRATION_DIR}/camera_calibration.npz"

# Attendance settings
ATTENDANCE_COOLDOWN = 60 * 60  # Seconds before a student can be marked present again (1 hour)

# Face recognition settings
FACE_RECOGNITION_MODEL = "VGG-Face"
FACE_RECOGNITION_METRIC = "cosine"
FACE_RECOGNITION_THRESHOLD = 0.95  # Minimum confidence for a valid face match

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CALIBRATION_DIR = "calibration_data"  # Directory to store calibration data
CALIBRATION_FILE = "calibration_data/camera_calibration.npz"  # File to save parameters
CHECKERBOARD_SIZE = (7, 6)  # 7 horizontal inner corners, 6 vertical
CALIBRATION_FRAMES_NEEDED = 10  # Number of good images needed

# UI Settings
FONT = "FONT_HERSHEY_SIMPLEX"
FONT_SCALE = 0.7
FONT_COLOR = (255, 255, 255)
RECT_COLOR = (0, 255, 0)
NEW_ATTENDANCE_COLOR = (0, 255, 0)
REGULAR_ATTENDANCE_COLOR = (0, 165, 255)