# README.txt

## Driver Drowsiness Detection System

This is a driver drowsiness detection system that uses a webcam to monitor a driver's facial features, detect signs of drowsiness, and send alerts via sound and Telegram messages.

### Requirements

1. Python 3.7 or higher
2. A laptop with a webcam
3. The following Python libraries:
   - OpenCV
   - Dlib
   - Numpy
   - Imutils
   - Playsound
   - Python Telegram Bot
   - Pytz

### Setup Instructions

1. **Install Required Libraries:**
   Open your command prompt or terminal and run:
   ```bash
   pip install opencv-python dlib numpy imutils playsound python-telegram-bot pytz

2. Prepare Necessary Files:
   Ensure the following files are in the same directory as your script:
      -drowsiness_detection.py (the main script)
      -shape_predictor_68_face_landmarks.dat (facial landmarks model)
      -haarcascade_frontalface_default.xml (Haar cascade for face detection)
      -haarcascade_eye.xml (Haar cascade for eye detection)
      -haarcascade_mcs_mouth.xml (Haar cascade for mouth detection)
      -alert.wav (sound file for alert)

3. Telegram Bot Setup:
   - Scan the QR code in the provided Telegram_bot.jpg image using your phone. 
     This will take you to Telegram to interact with the bot.
   
   - Send any message to the bot to start the interaction and it will send your chatID

   - Add the obtained chatID into the chat_ids section in the code and save the file.

4. Running the Script
   - Open your command prompt or terminal and navigate to the directory containing your script and necessary files.
   - Execute the script by running:  python drowsiness_detection.py

5. How It Works
   - The script captures video from the webcam.
   - It detects faces, eyes, and mouth using Haar cascades and Dlib's shape predictor.
   - It calculates the Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to determine if the eyes are closed or if the mouth is open (yawning).
   - If drowsiness is detected (closed eyes for a certain number of frames or yawning), an alert sound is played.
   - If drowsiness is detected three times, a Telegram message is sent to the specified chat IDs.

6. Notes
   - The script writes detection logs to detection_log.csv.
   - Press 'q' to quit the script.
   -Ensure that your webcam is properly connected and functioning.
   -Ensure all required files are in the same directory as the script.
   

May contact me at suthankanapathy@gmail.com for any doubts or clarifications.
