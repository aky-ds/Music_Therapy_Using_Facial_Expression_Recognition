from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pygame
import os

app = Flask(__name__)

# Load the model
model = load_model('model.h5')

# Initialize Pygame mixer
pygame.mixer.init()

# Emotion labels and music files
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}
music_files = {
    0: 'angry.mp3',
    1: 'disgust.mp3',
    2: 'fear.mp3',
    3: 'happy.mp3',
    4: 'neutral.mp3',
    5: 'sad.mp3',
    6: 'surprise.mp3',
}

# Variable to track the current playing emotion to prevent repeating music
current_emotion = None

# Preprocess image
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    return reshaped

# Predict emotion
def predict_emotion(img):
    preprocessed_img = preprocess_image(img)
    prediction = model.predict(preprocessed_img)
    emotion = np.argmax(prediction)
    return emotion

# Play music
def play_music(emotion):
    global current_emotion
    # Only play music if the emotion has changed
    if emotion != current_emotion:
        file = music_files.get(emotion)
        if file and os.path.exists(file):
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            current_emotion = emotion  # Update the current emotion

# Webcam frame generator
def generate_frames():
    camera = cv2.VideoCapture(0)  # Access the webcam
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Detect emotion
            emotion = predict_emotion(frame)

            # Overlay detected emotion on the frame
            cv2.putText(frame, f'Emotion: {emotion_labels.get(emotion, "Unknown")}', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Play music based on detected emotion
            play_music(emotion)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame to the browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    emotion = predict_emotion(img)
    play_music(emotion)
    return jsonify({
        'emotion': emotion_labels.get(emotion, 'Unknown'),
        'success': True
    })

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
