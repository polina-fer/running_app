import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import cv2
import mediapipe as mp
from collections import deque
import tempfile
import av

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseTransformer:
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.counter = 0
        self.display_counter = 0
        self.direction = "down"
        self.angle_history = deque(maxlen=15)
        self.rep_started = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(img, f"Reps: {self.display_counter}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Running Tracker")
st.write("Upload a video or start your webcam to track in real-time.")

col1, col2 = st.columns(2)
with col1:
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_video.read())
        tmp_path = tmp_file.name
    
    cap = cv2.VideoCapture(tmp_path)
    transformer = PoseTransformer()
    output_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        class DummyFrame:
            def __init__(self, img): self.img = img
            def to_ndarray(self, format): return self.img
            
        processed_frame_av = transformer.recv(DummyFrame(frame))
        processed_frame = processed_frame_av.to_ndarray(format="bgr24")
        output_frames.append(processed_frame)
    
    cap.release()
    
    if output_frames:
        out_path = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        height, width, _ = output_frames[0].shape
        out = cv2.VideoWriter(out_path, fourcc, 30.0, (width, height))
        for f in output_frames:
            out.write(f)
        out.release()
        st.video(out_path)
else:
    webrtc_streamer(
        key="pose-tracking", 
        video_processor_factory=PoseTransformer, 
        rtc_configuration=rtc_config, 
        media_stream_constraints={"video": True, "audio": False}
    )