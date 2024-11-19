import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def adjust_brightness(image, factor=1.5):
    """Adjust the brightness of the image by a given factor."""
    image = np.clip(image * factor, 0, 255).astype(np.uint8)
    return image

def analyze_face(image_path):
    try:
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Could not load image")

        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Flip the image horizontally
        flipped_image = cv2.flip(gray_image, 1)

        # Flip the image vertically
        vertically_flipped_image = cv2.flip(gray_image, 0)

        # Adjust brightness
        brightened_image = adjust_brightness(gray_image)

        # Detect facial landmarks
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            raise Exception("No face detected in the image")

        # Select 12 main keypoints
        key_points = [33, 133, 362, 263, 1, 61, 291, 199, 94, 0, 24, 130]

        height, width = gray_image.shape

        # Create a new figure for each analysis
        plt.clf()
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))  # Adjust to include 4 subplots

        # Plot the original image
        axes[0].imshow(gray_image, cmap='gray')
        axes[0].set_title('Imagen Original')
        for point_idx in key_points:
            landmark = results.multi_face_landmarks[0].landmark[point_idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            axes[0].plot(x, y, 'rx')

        # Plot the flipped image horizontally
        axes[1].imshow(flipped_image, cmap='gray')
        axes[1].set_title('Imagen Horizontalmente')
        for point_idx in key_points:
            landmark = results.multi_face_landmarks[0].landmark[point_idx]
            x = width - int(landmark.x * width)  # Flip x-coordinate
            y = int(landmark.y * height)
            axes[1].plot(x, y, 'rx')

        # Plot the brightened image
        axes[2].imshow(brightened_image, cmap='gray')
        axes[2].set_title('Brillo Aumentado')
        for point_idx in key_points:
            landmark = results.multi_face_landmarks[0].landmark[point_idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            axes[2].plot(x, y, 'rx')

        # Plot the flipped image vertically
        axes[3].imshow(vertically_flipped_image, cmap='gray')
        axes[3].set_title('Imagen Verticalmente')
        for point_idx in key_points:
            landmark = results.multi_face_landmarks[0].landmark[point_idx]
            x = int(landmark.x * width)  # X-coordinate remains the same
            y = height - int(landmark.y * height)  # Flip y-coordinate
            axes[3].plot(x, y, 'rx')

        # Save plot to memory
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        # Convert to base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64

    except Exception as e:
        print(f"Error in analyze_face: {str(e)}")
        raise
    finally:
        plt.close('all')

@app.route('/')
def home():
    # Get list of images in upload folder
    images = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            images.append(filename)
    return render_template('index.html', images=images)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if we're analyzing an existing file
        if 'existing_file' in request.form:
            filename = request.form['existing_file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                return jsonify({'error': f'File not found: {filename}'}), 404
            
        # Check if we're uploading a new file
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        
        else:
            return jsonify({'error': 'No file provided'}), 400

        # Analyze the image
        result_image = analyze_face(filepath)
        
        return jsonify({
            'success': True,
            'image': result_image
        })

    except Exception as e:
        print(f"Error in /analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
