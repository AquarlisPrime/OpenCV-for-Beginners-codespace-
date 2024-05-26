import cv2
import mediapipe as mp

# Define colors for drawing
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)

# Define connections for drawing skeleton lines and finger connections
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
FINGER_CONNECTIONS = [
    (4, 3), (3, 2), (2, 1),   # Thumb
    (8, 7), (7, 6), (6, 5),   # Index finger
    (12, 11), (11, 10), (10, 9),   # Middle finger
    (16, 15), (15, 14), (14, 13),  # Ring finger
    (20, 19), (19, 18), (18, 17)   # Little finger
]

# Open video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Pose model and drawing utils
with mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Draw skeleton lines
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, POSE_CONNECTIONS)

            # Draw finger connections
            for connection in FINGER_CONNECTIONS:
                landmark_from, landmark_to = connection
                if all(results.pose_landmarks.landmark[landmark].visibility > 0 for landmark in [landmark_from, landmark_to]):
                    x1 = int(results.pose_landmarks.landmark[landmark_from].x * frame.shape[1])
                    y1 = int(results.pose_landmarks.landmark[landmark_from].y * frame.shape[0])
                    x2 = int(results.pose_landmarks.landmark[landmark_to].x * frame.shape[1])
                    y2 = int(results.pose_landmarks.landmark[landmark_to].y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), COLOR_BLUE, 3)

        # Display the annotated frame
        cv2.imshow('Custom Pose Detector', frame)

        # Check for key press events
        key = cv2.waitKey(1)

        # Exit loop if 'q' key is pressed
        if key & 0xFF == ord('q'):
            break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
 
