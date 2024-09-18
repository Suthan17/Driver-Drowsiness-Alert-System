import cv2
import dlib
import numpy as np
from imutils import face_utils
from playsound import playsound
import datetime
import uuid
from telegram import Bot
import asyncio
import pytz

# Telegram configuration
bot_token = '7465441789:AAEizYiYqlkRQF9N9hH_76UwllW30VzlgnQ'
chat_ids = ['981927909','831625286']  # Replace with your actual chat ID

bot = Bot(token=bot_token)

async def send_alert_message():
    for chat_id in chat_ids:
        await bot.send_message(chat_id=chat_id, text="The driver is drowsy. Please ask them to take a rest.")
        print(f"Message sent to {chat_id}")

# Load the cascades for face, eye, and mouth detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')  # Ensure this file is in the same directory

# Initialize the dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Ensure this file is in the same directory

# 3D model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Open log file
log_file = open("detection_log.csv", "a")

# Write the header to the log file
log_file.write("DetectionID,DetectionTime,DrowsinessLevel,FrameID,Timestamp,EyeClosureMetric,EAR,MouthAspectMetric,MAR,HeadPose,AlarmTime\n")

# Parameters for drowsiness detection
closed_eyes_frames = 0
alert_threshold = 20  # Number of frames with closed eyes to trigger alert
yawn_frames = 0
yawn_threshold = 15  # Number of frames with mouth open to trigger alert
frame_id = 0
alert_count = 0

# Malaysian timezone
malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')

async def main_loop():
    global closed_eyes_frames, yawn_frames, frame_id, alert_count
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Increment frame ID
        frame_id += 1

        # Convert frame to grayscale for face and eye detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = frame.shape

        # Detect faces in the frame
        faces = detector(gray)
        for face in faces:
            detection_id = uuid.uuid4()  # Unique ID for each detection
            detection_time = datetime.datetime.now(malaysia_tz)  # Malaysian time
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue box for face

            # Log face detection
            log_file.write(f"{detection_id},{detection_time},,,{frame_id},{detection_time},,,,\n")

            # Get facial landmarks
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # Extract eye regions
            left_eye = shape[36:42]
            right_eye = shape[42:48]

            # Check if eyes are closed
            left_eye_ratio = np.mean(np.linalg.norm(left_eye[1] - left_eye[5]) + np.linalg.norm(left_eye[2] - left_eye[4])) / (2.0 * np.linalg.norm(left_eye[0] - left_eye[3]))
            right_eye_ratio = np.mean(np.linalg.norm(right_eye[1] - right_eye[5]) + np.linalg.norm(right_eye[2] - right_eye[4])) / (2.0 * np.linalg.norm(right_eye[0] - right_eye[3]))
            eye_aspect_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

            if eye_aspect_ratio < 0.25:  # Threshold for eye aspect ratio indicating closed eyes
                closed_eyes_frames += 1
            else:
                closed_eyes_frames = 0

            for (ex, ey) in left_eye:
                cv2.circle(frame, (ex, ey), 1, (0, 255, 0), -1)  # Green points for left eye
            for (ex, ey) in right_eye:
                cv2.circle(frame, (ex, ey), 1, (0, 255, 0), -1)  # Green points for right eye

            # Log eye detection
            log_file.write(f"{detection_id},{detection_time},,{frame_id},{detection_time},,{eye_aspect_ratio},,,\n")

            # Detect mouth region using mouth cascade
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            mouth_rects = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))

            mar = 0  # Mouth Aspect Ratio
            for (mx, my, mw, mh) in mouth_rects:
                yawn_frames += 1
                cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 255, 255), 2)  # Yellow box for mouth
                mar = mh / mw  # Simple aspect ratio for mouth (could be refined)
                log_file.write(f"{detection_id},{detection_time},,{frame_id},{detection_time},,,,{mar},\n")
                break  # Only consider the first detected mouth
            
            if len(mouth_rects) == 0:
                yawn_frames = 0

            # Head pose estimation
            image_points = np.array([
                shape[30],     # Nose tip
                shape[8],      # Chin
                shape[36],     # Left eye left corner
                shape[45],     # Right eye right corner
                shape[48],     # Left mouth corner
                shape[54]      # Right mouth corner
            ], dtype="double")

            focal_length = size[1]
            center = (size[1]/2, size[0]/2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]], dtype="double"
            )
            dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion

            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
            (nose_end_point2D, _) = cv2.projectPoints(np.array([(0, 0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            
            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            cv2.line(frame, p1, p2, (255, 0, 0), 2)  # Blue line for head pose direction
            head_pose = (p1, p2)
            log_file.write(f"{detection_id},{detection_time},,{frame_id},{detection_time},,,,{mar},{head_pose}\n")

            # Check for drowsiness or yawning
            if closed_eyes_frames >= alert_threshold or yawn_frames >= yawn_threshold:
                playsound('alert.wav')  # Ensure you have an alert.wav file
                alarm_time = datetime.datetime.now(malaysia_tz)  # Malaysian time
                alert_count += 1
                log_file.write(f"{detection_id},{detection_time},,{frame_id},{detection_time},,,,,,{alarm_time}\n")

                # Send alert message if detected three times
                if alert_count >= 3:
                    await send_alert_message()
                    alert_count = 0  # Reset alert count after sending message

                closed_eyes_frames = 0  # Reset the counter after alert
                yawn_frames = 0  # Reset the counter after alert

        # Display the resulting frame
        cv2.imshow('Webcam', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Run the main loop
loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(main_loop())
finally:
    loop.close()

# Close log file
log_file.close()

# When everything is done, release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
