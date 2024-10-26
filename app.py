import os
import re
from flask import Flask, request, render_template
import cv2
import mediapipe as mp
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def sanitize_filename(filename):
    # Reemplazar caracteres no permitidos con un guion bajo
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

def draw_cross(image, center, size=5, color=(0, 0, 255)):
    """Dibuja una cruz en la imagen en la posición especificada."""
    x, y = center
    cv2.line(image, (x - size, y - size), (x + size, y + size), color, 2)  # Diagonal de izquierda a derecha
    cv2.line(image, (x + size, y - size), (x - size, y + size), color, 2)  # Diagonal de derecha a izquierda

def process_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    facial_points_dict = {
        'left_eye_center_x': [None], 'left_eye_center_y': [None],
        'right_eye_center_x': [None], 'right_eye_center_y': [None],
        'left_eye_inner_corner_x': [None], 'left_eye_inner_corner_y': [None],
        'left_eye_outer_corner_x': [None], 'left_eye_outer_corner_y': [None],
        'right_eye_inner_corner_x': [None], 'right_eye_inner_corner_y': [None],
        'right_eye_outer_corner_x': [None], 'right_eye_outer_corner_y': [None],
        'left_eyebrow_inner_end_x': [None], 'left_eyebrow_inner_end_y': [None],
        'left_eyebrow_outer_end_x': [None], 'left_eyebrow_outer_end_y': [None],
        'right_eyebrow_inner_end_x': [None], 'right_eyebrow_inner_end_y': [None],
        'right_eyebrow_outer_end_x': [None], 'right_eyebrow_outer_end_y': [None],
        'nose_tip_x': [None], 'nose_tip_y': [None],
        'mouth_left_corner_x': [None], 'mouth_left_corner_y': [None],
        'mouth_right_corner_x': [None], 'mouth_right_corner_y': [None],
        'mouth_center_top_lip_x': [None], 'mouth_center_top_lip_y': [None],
        'mouth_center_bottom_lip_x': [None], 'mouth_center_bottom_lip_y': [None],
        'Image': [image_path]
    }

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks_mapping = {
                33: 'left_eye_center', 263: 'right_eye_center',
                133: 'left_eye_inner_corner', 362: 'right_eye_inner_corner',
                130: 'left_eye_outer_corner', 359: 'right_eye_outer_corner',
                55: 'left_eyebrow_inner_end', 285: 'right_eyebrow_inner_end',
                105: 'left_eyebrow_outer_end', 334: 'right_eyebrow_outer_end',
                1: 'nose_tip', 61: 'mouth_left_corner', 291: 'mouth_right_corner',
                0: 'mouth_center_top_lip', 17: 'mouth_center_bottom_lip'
            }

            for idx, landmark in enumerate(face_landmarks.landmark):
                # Obtener coordenadas escaladas
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])

                # Verificar si el índice está en nuestro mapeo
                if idx in landmarks_mapping:
                    key_x = f"{landmarks_mapping[idx]}_x"
                    key_y = f"{landmarks_mapping[idx]}_y"
                    facial_points_dict[key_x][0] = x
                    facial_points_dict[key_y][0] = y
                    
                    # Dibuja una cruz roja en los puntos seleccionados
                    draw_cross(image, (x, y))

    # Guardar la imagen procesada
    output_image_path = os.path.splitext(image_path)[0] + '_processed.jpeg'
    cv2.imwrite(output_image_path, image)

    return pd.DataFrame(facial_points_dict), output_image_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            sanitized_filename = sanitize_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], sanitized_filename)
            file.save(file_path)

            df_facial_points, processed_image_path = process_image(file_path)
            output_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'facial_points.csv')
            df_facial_points.to_csv(output_csv_path, index=False)

            return render_template('result.html', image=processed_image_path, csv_file='facial_points.csv')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
