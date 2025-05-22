import cv2
import mediapipe as mp
import copy
import random
import pygame
from mediapipe.framework.formats import landmark_pb2

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize webcam
cap = cv2.VideoCapture(0)

# Sparkle effect function
def draw_sparkle(image, x, y):
    color = (random.randint(150, 255), random.randint(150, 255), 255)
    cv2.circle(image, (x, y), 5, color, -1)

# Global settings
current_color = (0, 255, 255)
current_radius = 6
clone_trail = []
mirror_gap = 0.0
gap_direction = 1  # for auto-bounce effect

# Start background music after webcam is ready
pygame.mixer.init()
pygame.mixer.music.load("background.mp3.mp3")  # Make sure this file exists in the same folder
pygame.mixer.music.play(-1)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        height, width, _ = image.shape

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            mirrored_landmarks = []

            for lm in landmarks:
                mirrored = copy.deepcopy(lm)
                mirrored.x = min(max(1.0 - lm.x + mirror_gap, 0.0), 1.0)
                mirrored.y = lm.y
                mirrored_landmarks.append(mirrored)

            clone = landmark_pb2.NormalizedLandmarkList(landmark=mirrored_landmarks)
            clone_trail.append(clone)
            if len(clone_trail) > 10:
                clone_trail.pop(0)

            # Animation trails
            for i, past_pose in enumerate(clone_trail):
                fade_color = tuple([int(c * (1 - i / 10.0)) for c in current_color])
                mp_drawing.draw_landmarks(
                    image, past_pose, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=fade_color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                )

            # Current clone
            mp_drawing.draw_landmarks(
                image, clone, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=current_color, thickness=4, circle_radius=current_radius),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4)
            )

            # Sparkles
            for lm in mirrored_landmarks:
                cx, cy = int(lm.x * width), int(lm.y * height)
                draw_sparkle(image, cx, cy)

            # Gesture controls
            rw = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            lw = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

            # Right hand raised – Change color
            if rw.y < nose.y:
                current_color = (
                    random.randint(100, 255),
                    random.randint(100, 255),
                    random.randint(100, 255)
                )

            # Left hand raised – Adjust clone gap
            if lw.y < nose.y:
                mirror_gap += 0.01 * gap_direction
                if mirror_gap > 0.3 or mirror_gap < -0.3:
                    gap_direction *= -1  # reverse direction (bounce)

        cv2.imshow('Mirrored Neon Clone Pose', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            pygame.mixer.music.stop()

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
