import cv2
import mediapipe as mp
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5
)

@app.route('/')
def home():
    return 'Welcome to the render.com server'

@app.route('/process_image', methods=['POST'])
def process_image():
    # Check if the POST request has a file part
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image']

    # If the user does not select a file, the browser may submit an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file is an allowed extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
        # Save the file to a temporary location
        file_path = 'temp_image.jpg'
        file.save(file_path)

        # Read the image
        image = cv2.imread(file_path)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Check if left hand is above or below the left shoulder
        left_wrist_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height
        left_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height

        if left_wrist_y > left_shoulder_y:
            return jsonify({'result': 'Left hand is above the left shoulder'})
        else:
            return jsonify({'result': 'Left hand is below the left shoulder'})

    else:
        return jsonify({'error': 'Invalid file extension'})


if __name__ == '__main__':
    app.run(host='0.0.0.0')
