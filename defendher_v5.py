from flask import Flask, Response, render_template, jsonify, request,json
import cv2
import torch
import mediapipe as mp
from datetime import datetime
import os
import time
import sqlite3

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv5 model for person detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

# MediaPipe for hand gesture detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Load gender classification model
gender_model = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
GENDER_LIST = ['Male', 'Female']

# SOS Alerts Storage
sos_alerts = []

# Function to predict gender
def predict_gender(face_image):
    try:
        blob = cv2.dnn.blobFromImage(face_image, 1.0, (227, 227),
                                      (78.4263377603, 87.7689143744, 114.895847746),
                                      swapRB=False)
        gender_model.setInput(blob)
        gender_preds = gender_model.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]
        return gender
    except Exception as e:
        print(f"Error predicting gender: {e}")
        return "Unknown"

# Detect SOS gesture
def detect_sos_gesture(hand_landmarks):
    tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]

    mcps = [
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]

    thumb_tip = hand_landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks[mp_hands.HandLandmark.THUMB_MCP]
    wrist = hand_landmarks[mp_hands.HandLandmark.WRIST]

    # Check if fingers are closed
    fingers_closed = all(
        tip.y > mcp.y for tip, mcp in zip(
            [hand_landmarks[tip] for tip in tips],
            [hand_landmarks[mcp] for mcp in mcps]
        )
    )

    # Check if thumb is tucked
    thumb_tucked = (
        thumb_tip.x > thumb_mcp.x and
        thumb_tip.y > wrist.y
    )

    return fingers_closed and thumb_tucked

# Process video frame
# Function to save SOS alert to the database
def save_sos_alert(timestamp, details, location=None):
    conn = sqlite3.connect('sos_alerts.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO sos_alerts (timestamp, details, location)
        VALUES (?, ?, ?)
    ''', (timestamp, details, location))
    conn.commit()
    conn.close()

# Modified part of the process_frame function to save alerts
def process_frame():
    ret, frame = cap.read()
    if not ret:
        return None, 0, 0, False

    # YOLOv5 detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    male_count = 0
    female_count = 0
    sos_triggered = False

    for *box, conf, cls in detections:
        label = results.names[int(cls)]
        if label == 'person' and conf > 0.5:
            x1, y1, x2, y2 = map(int, box)
            
            # Crop the person region
            person_crop = frame[y1:y2, x1:x2]
            
            # Classify gender
            try:
                gender = predict_gender(person_crop)
            except Exception as e:
                gender = "Unknown"
                print(f"Error predicting gender: {e}")

            # Count genders
            if gender == "Male":
                male_count += 1
            elif gender == "Female":
                female_count += 1
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{gender}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Convert frame for gesture detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    # Detect SOS gesture
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Convert landmarks to list for easier indexing
            landmarks_list = [
                hand_landmarks.landmark[i] for i in range(len(mp_hands.HandLandmark))
            ]
            
            if detect_sos_gesture(landmarks_list):
                sos_triggered = True
                # Record SOS alert
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                location = "Lat: 9°01'30.6\"N, Lng: 72°51'01.1\"E"  # Placeholder location, this can be dynamic
                details = 'SOS Gesture Detected'

                # Save the SOS alert to the database
                save_sos_alert(timestamp, details, location)

                # Optionally, you can also keep it in memory for real-time updates
                alert = {
                    'timestamp': timestamp,
                    'details': details,
                    'location': location,
                    'male_count': male_count,
                    'female_count': female_count
                }
                sos_alerts.append(alert)

    return frame, male_count, female_count, sos_triggered

# Flask routes
@app.route('/video_feed')
def video_feed():
    def generate_video():
        while True:
            frame, _, _, _ = process_frame()
            if frame is None:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate_video(), content_type='multipart/x-mixed-replace; boundary=frame')

@app.route('/events')
def events():
    def event_stream():
        while True:
            _, male_count, female_count, sos_triggered = process_frame()
            update_message = {
                    "male_count": male_count,
                    "female_count": female_count,
                    "sos_triggered": sos_triggered,
                    "alerts": sos_alerts if sos_triggered else [],
                }
            yield f"data: {json.dumps(update_message)}\n\n"
            time.sleep(1)

    return Response(event_stream(), content_type='text/event-stream')

@app.route('/sos_dashboard')
def sos_dashboard():
    # Connect to the database
    conn = sqlite3.connect('sos_alerts.db')
    c = conn.cursor()

    # Fetch all SOS alerts from the database
    c.execute('SELECT * FROM sos_alerts ORDER BY timestamp DESC')
    alerts = c.fetchall()

    # Categorize the alerts based on details
    gesture_sos = sum(1 for alert in alerts if 'SOS Gesture Detected' in alert[2])
    lone_women_sos = sum(1 for alert in alerts if 'Lone Women SOS' in alert[2])
    armed_sos = sum(1 for alert in alerts if 'Armed SOS' in alert[2])
    other = len(alerts) - gesture_sos - lone_women_sos - armed_sos

    # Close the database connection
    conn.close()

    # Pass categorized data to template
    return render_template('sos_dashboard.html', 
                           alerts=alerts, 
                           gesture_sos=gesture_sos, 
                           lone_women_sos=lone_women_sos, 
                           armed_sos=armed_sos,
                           other=other)
@app.route('/get_sos_alerts')

@app.route('/mobile_view')
def mobile_view():
    # Connect to the database
    conn = sqlite3.connect('sos_alerts.db')
    c = conn.cursor()

    # Fetch all SOS alerts from the database
    c.execute('SELECT * FROM sos_alerts ORDER BY timestamp DESC')
    alerts = c.fetchall()

    # Close the database connection
    conn.close()

    # Pass SOS alerts data to the mobile view template
    return render_template('mobile_view.html', alerts=alerts)

def get_sos_alerts():
    return jsonify(sos_alerts)

@app.route('/trigger_user_sos', methods=['POST'])
def trigger_user_sos():
    user_id = request.json.get('user_id')
    if user_id:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        details = 'User Triggered SOS'
        location = request.json.get('location', "Unknown")
        
        save_sos_alert(timestamp, details, location)
        return jsonify({"message": "User SOS triggered successfully"})
    else:
        return jsonify({"error": "User ID is required"}), 400

@app.route('/log_sos_location', methods=['POST'])
def log_sos_location():
    data = request.json
    if data and 'location' in data:
        latitude = data['location']['latitude']
        longitude = data['location']['longitude']
        if sos_alerts:
            sos_alerts[-1]['location'] = f"Lat: {latitude}, Lng: {longitude}"
    return '', 204

@app.route('/')
def index():
    return render_template('index.html')

# Template files
templates_dir = os.path.join(os.path.dirname(__file__), 'templates')

@app.route('/get_emergency_contact/<int:user_id>', methods=['GET'])
def get_emergency_contact(user_id):
    # Connect to the SQLite database
    conn = sqlite3.connect('sos_alerts.db')
    c = conn.cursor()

    # Query the users table for emergency contact details
    c.execute('SELECT emergency_1, emergency_no FROM users WHERE id = ?', (user_id,))
    user_data = c.fetchone()

    # If user found, return the data as JSON
    if user_data:
        emergency_contact = {
            'emergency_contact': user_data[0],
            'emergency_number': user_data[1]
        }
        return jsonify(emergency_contact)
    else:
        return jsonify({'error': 'User not found'}), 404





# Run Flask app
if __name__ == "__main__":
    app.run(debug=True, threaded=True, host="0.0.0.0", port=5000)
