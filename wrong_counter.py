import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import av
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import pygame

########################Video Capture###################
mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
count = 0
max_wrong_count = 4

# Function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Custom VideoTransformer class
class PoseVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5
        )
        self.level = None
        self.wrong_count = 0

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Rest of the code...
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Making detection
        res = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ########################################
        # extract landmarks
        try:
            landmarks = res.pose_landmarks.landmark
            # Get coordinates
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder_r, elbow_R, wrist_R)

            # Visualize angle
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow_R, [780, 240]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Curl counter logic
            if angle > 160:
                self.level = "down"
            if angle < 45 and self.level == 'down':
                self.level = "up"
                global count
                count += 1
                self.wrong_count = 0  # Reset wrong count after correct repetition
                st.write(f"Count: {count}")
            else:
                self.wrong_count += 1  # Increment wrong count for wrong repetition
                if self.wrong_count == max_wrong_count:
                    # Play sound for 4 consecutive wrong repetitions
                    pygame.mixer.init()
                    pygame.mixer.music.load('wrong_rep_sound.mp3')
                    pygame.mixer.music.play()

            # Draw landmarks
            mp_draw.draw_landmarks(image, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        except:
            pass

        cv2.putText(image, 'COUNT', (20, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(count),
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'LEVEL', (100, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, self.level,
                    (95, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def main():
    # Streamlit application code...
    st.title("Fitness Tracker")
    st.write("Real-time Video Feed")

    webrtc_streamer(
        key="pose",
        video_transformer_factory=PoseVideoTransformer,
        async_transform=True,
    )

if __name__ == '__main__':
    main()