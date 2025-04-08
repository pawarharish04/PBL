from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from scipy.spatial.distance import cdist
import os
import base64
from threading import Thread
import time

app = Flask(__name__, static_folder="static", template_folder="templates")

# Global variables
camera = None
processing_active = False

# Constants
MODEL_FILE = "hand_gesture_model.tflite"
NORMALIZER_FILE = "normalizer_params.npz"
CLASS_NAMES_FILE = "class_names.npy"
BASE_DISTANCES = 210

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

class FeatureNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1

    def transform(self, X):
        return (X - self.mean) / self.std

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        normalizer = cls()
        normalizer.mean = data['mean']
        normalizer.std = data['std']
        return normalizer

# Load model and normalizer
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    normalizer = FeatureNormalizer.load(NORMALIZER_FILE)
    class_names = np.load(CLASS_NAMES_FILE, allow_pickle=True)
    print("Model and supporting files loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")

def draw_prediction_overlay(frame, class_name, confidence):
    """Draw an advanced prediction box on top-left corner"""
    overlay_text = f"{class_name}"
    confidence_text = f"{confidence * 100:.1f}%"

    x, y, w, h = 10, 10, 200, 60
    box_color = (30, 30, 30)
    text_color = (255, 255, 255)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.putText(frame, overlay_text, (x + 15, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)

    cv2.putText(frame, confidence_text, (x + 15, y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
    return frame

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            dist_matrix = cdist(landmarks, landmarks, 'euclidean')
            features = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

            if len(features) == BASE_DISTANCES:
                features = normalizer.transform(features[np.newaxis, :]).astype(np.float32)
                interpreter.set_tensor(input_details[0]["index"], features)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]["index"])[0]
                class_id = np.argmax(predictions)
                confidence = float(predictions[class_id])
                class_name = str(class_names[class_id])
                frame = draw_prediction_overlay(frame, class_name, confidence)

    return frame

def generate_frames():
    global camera, processing_active

    while processing_active:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        processed_frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    gestures_folder = os.path.join(app.static_folder, 'abcd')
    gestures = []

    if os.path.exists(gestures_folder):
        for folder_name in os.listdir(gestures_folder):
            folder_path = os.path.join(gestures_folder, folder_name)
            if os.path.isdir(folder_path):
                images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if images:
                    gestures.append({
                        'name': folder_name.capitalize(),
                        'folder': folder_name,
                        'image': images[0]
                    })

    return render_template('index.html', gestures=gestures)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/documentation')
def documentation():
    return render_template('documentation.html')





@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    global camera, processing_active

    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    processing_active = True
    return jsonify({"status": "started", "message": "Recognition started successfully"})

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    global camera, processing_active

    processing_active = False

    if camera is not None:
        camera.release()
        camera = None

    return jsonify({"status": "stopped", "message": "Recognition stopped successfully"})

@app.route('/get_gesture_images')
def get_gesture_images():
    image_folder = os.path.join(app.static_folder, 'abcd')
    image_files = [
        f'abcd/{filename}' for filename in os.listdir(image_folder)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))
    ]
    return jsonify({"images": image_files})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image provided"})

    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "No image selected"})

    try:
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.6
        ) as static_hands:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = static_hands.process(rgb_image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    dist_matrix = cdist(landmarks, landmarks, 'euclidean')
                    features = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

                    if len(features) == BASE_DISTANCES:
                        features = normalizer.transform(features[np.newaxis, :]).astype(np.float32)
                        interpreter.set_tensor(input_details[0]["index"], features)
                        interpreter.invoke()
                        predictions = interpreter.get_tensor(output_details[0]["index"])[0]
                        class_id = np.argmax(predictions)
                        confidence = float(predictions[class_id])
                        class_name = str(class_names[class_id])

                        cv2.putText(image, f"{class_name} ({confidence:.2f})", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                cv2.putText(image, "No hand detected", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', image)
        img_str = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "success": True,
            "image": f"data:image/jpeg;base64,{img_str}"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)