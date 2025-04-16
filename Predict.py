import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from model import load_model  
import tensorflow as tf

st.set_page_config(page_title="GazeFlow - Predict", layout="centered")
st.title("Predict Gaze Direction")

model = load_model()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

LEFT_EYE_INDICES = [33, 133, 160, 158, 144, 145, 153, 154, 155]
RIGHT_EYE_INDICES = [263, 362, 387, 385, 373, 374, 380, 381, 382]

def crop_eye(frame, landmarks, indices, padding_ratio=0.5, target_size=(120, 80)):
    h, w, _ = frame.shape
    points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indices]

    x_min = min(p[0] for p in points)
    y_min = min(p[1] for p in points)
    x_max = max(p[0] for p in points)
    y_max = max(p[1] for p in points)

    box_width = x_max - x_min
    box_height = y_max - y_min
    pad_x = int(box_width * padding_ratio)
    pad_y = int(box_height * padding_ratio)

    x_min = max(x_min - pad_x, 0)
    y_min = max(y_min - pad_y, 0)
    x_max = min(x_max + pad_x, w)
    y_max = min(y_max + pad_y, h)

    cropped = frame[y_min:y_max, x_min:x_max]
    
    cropped_resized = cv2.resize(cropped, target_size)
    return cropped_resized

def preprocess_eye(img):
    img = tf.keras.preprocessing.image.img_to_array(img)
    return img

def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

st.info("**Before capturing the photo:**\n- Maximize your screen brightness\n- Come a bit closer to the camera\n- Remove your spectacles for better accuracy")
uploaded_image = st.camera_input("Take a photo to predict gaze direction")

if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        left_eye_img = crop_eye(frame, face_landmarks, LEFT_EYE_INDICES)
        right_eye_img = crop_eye(frame, face_landmarks, RIGHT_EYE_INDICES)

        left_eye_img_rgb = cv2.cvtColor(left_eye_img, cv2.COLOR_BGR2RGB)
        right_eye_img_rgb = cv2.cvtColor(right_eye_img, cv2.COLOR_BGR2RGB)

        pre_left = preprocess_eye(left_eye_img_rgb)
        pre_right = preprocess_eye(right_eye_img_rgb)

        left_pred = model.predict(np.expand_dims(pre_left, axis=0))[0]
        right_pred = model.predict(np.expand_dims(pre_right, axis=0))[0]

        st.markdown(f"""
        ### Gaze Predictions  
        - **Left Eye**: `{np.round(left_pred, 2)}`  
        - **Right Eye**: `{np.round(right_pred, 2)}`
        """)

        h, w, _ = frame.shape

        def get_eye_center(landmarks, indices):
            x_center = int(np.mean([landmarks.landmark[i].x for i in indices]) * w)
            y_center = int(np.mean([landmarks.landmark[i].y for i in indices]) * h)
            return (x_center, y_center)

        left_center = get_eye_center(face_landmarks, LEFT_EYE_INDICES)
        right_center = get_eye_center(face_landmarks, RIGHT_EYE_INDICES)

        arrow_scale = 50  
        left_vector = normalize_vector(left_pred[:2]) * arrow_scale
        right_vector = normalize_vector(right_pred[:2]) * arrow_scale

        left_end_point = (int(left_center[0] + left_vector[0]), int(left_center[1] + left_vector[1]))
        right_end_point = (int(right_center[0] + right_vector[0]), int(right_center[1] + right_vector[1]))

        cv2.arrowedLine(frame, left_center, left_end_point, (0, 255, 255), thickness=2)
        cv2.arrowedLine(frame, right_center, right_end_point, (0, 255, 255), thickness=2)

        st.image(frame[:, :, ::-1], caption="Gaze Direction", use_container_width=True)  

