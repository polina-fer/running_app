import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import cv2
import mediapipe as mp
from collections import deque
import tempfile
import av
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

class PoseTransformer:
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Inicijalizacija brojača i stanja za desnu nogu
        self.right_counter = 0
        self.right_direction = "up"
        self.right_history = deque(maxlen=15)
        self.right_started = False
        
        # Inicijalizacija brojača i stanja za lijevu nogu
        self.left_counter = 0
        self.left_direction = "up"
        self.left_history = deque(maxlen=15)
        self.left_started = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Desna noga
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
            
            r_angle = calculate_angle(r_hip, r_knee, r_ankle)
            self.right_history.append(r_angle)

            # Povijest kutova za određivanje ekstrema
            if len(self.right_history) == 15:
                r_mid = self.right_history[7]
                r_min = all(r_mid < self.right_history[i] for i in range(15) if i != 7)
                r_max = all(r_mid > self.right_history[i] for i in range(15) if i != 7)

                # Noga se diže - traži se minimum, ponavljanje počinje
                if self.right_direction == "up":
                    if r_min and not self.right_started and r_mid < 110:
                        self.right_started = True
                        self.right_direction = "down"
                
                # Noga se spušta - traži se maksimum, ponavljanje završava
                elif self.right_direction == "down":
                    if r_max and self.right_started and r_mid > 150:
                        self.right_started = False
                        self.right_counter += 1
                        self.right_direction = "up"

            # Lijeva noga
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
            
            l_angle = calculate_angle(l_hip, l_knee, l_ankle)
            self.left_history.append(l_angle)

            # Povijest kutova za određivanje ekstrema
            if len(self.left_history) == 15:
                l_mid = self.left_history[7]
                l_min = all(l_mid < self.left_history[i] for i in range(15) if i != 7)
                l_max = all(l_mid > self.left_history[i] for i in range(15) if i != 7)

                # Noga se diže - traži se minimum, ponavljanje počinje
                if self.left_direction == "up":
                    if l_min and not self.left_started and l_mid < 110:
                        self.left_started = True
                        self.left_direction = "down"

                # Noga se spušta - traži se maksimum, ponavljanje završava
                elif self.left_direction == "down":
                    if l_max and self.left_started and l_mid > 150:
                        self.left_started = False
                        self.left_counter += 1
                        self.left_direction = "up"

            # Crteži i vizualizacija
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Prikaz brojača
            cv2.putText(img, f"Right: {self.right_counter}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f"Left: {self.left_counter}", (50, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Prikaz ukupnih koraka
            cv2.putText(img, f"Total: {self.left_counter + self.right_counter}", (50, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Running Tracker")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_video.read())
        tmp_path = tmp_file.name
    
        cap = cv2.VideoCapture(tmp_path)
        transformer = PoseTransformer()
        frame_placeholder = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            class DummyFrame:
                def __init__(self, img): self.img = img
                def to_ndarray(self, format): return self.img
                
            processed_frame_av = transformer.recv(DummyFrame(frame))
            processed_frame = processed_frame_av.to_ndarray(format="bgr24")
            frame_placeholder.image(processed_frame, channels="BGR")
        
        cap.release()
else:
    webrtc_streamer(
        key="pose-tracking", 
        video_processor_factory=PoseTransformer, 
        rtc_configuration=rtc_config, 
        media_stream_constraints={
            "video": {"width": 480, "height": 360, "frameRate": 15},
            "audio": False
        },
        async_processing=True
    )