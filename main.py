import os
import time
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import os
import shutil
import tensorflow as tf
from keras.models import load_model
app = Flask(__name__)

#model
class PatchExtractor(tf.keras.layers.Layer):
    def __init__(self):
        super(PatchExtractor, self).__init__()

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, 48, 48, 1],
            strides=[1, 48, 48, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, 4, patch_dims])
        return patches


model = load_model(r'model/checkpoint_epoch_83.h5', custom_objects={'PatchExtractor': PatchExtractor})

recording = False
output_folders = []
current_folder_index = 0
start_time = None
first_run = True  # Variable de drapeau pour indiquer si c'est le premier démarrage de l'application

# Load the SSD model for face detection
net = cv2.dnn.readNetFromCaffe('cascade/deploy.prototxt', 'cascade/modeldetect.caffemodel')

# Fonction pour vider les dossiers d'enregistrement et supprimer les vidéos
# Fonction pour vider les dossiers d'enregistrement et supprimer les vidéos
def cleanup_folders():
    # Supprimer les vidéos
    for i in range(4):
        video_path = os.path.join('recordings', f"video_{i}.mp4")
        if os.path.exists(video_path):
            os.remove(video_path)

    # Vider les dossiers d'enregistrement
    for folder in output_folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    # Vider les dossiers folderRet_x
    for i in range(4):
        folder_ret_path = os.path.join('recordings', f"folderRet_{i}")
        if os.path.exists(folder_ret_path):
            for filename in os.listdir(folder_ret_path):
                file_path = os.path.join(folder_ret_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

# Création des dossiers pour stocker les enregistrements
base_output_path = 'recordings'
if not os.path.exists(base_output_path):
    os.makedirs(base_output_path)

for i in range(4):
    folder_path = os.path.join(base_output_path, f"folder_{i}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    output_folders.append(folder_path)

    folder_ret_path = os.path.join(base_output_path, f"folderRet_{i}")
    if not os.path.exists(folder_ret_path):
        os.makedirs(folder_ret_path)
# Nettoyer les dossiers d'enregistrement uniquement au premier démarrage de l'application
if first_run:
    cleanup_folders()
    first_run = False

# Route principale pour afficher la page HTML
@app.route('/')
def index():
    return render_template('index.html')

# Fonction pour détecter les visages
def detect_faces(frame):
    # Detect faces using SSD
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    face_boxes = []
    # Extract faces from the detected faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, w, h) = box.astype("int")
            face_boxes.append((x, y, w, h))

    return face_boxes

# Fonction pour générer les frames à partir du flux vidéo
def generate_frames():
    global current_folder_index, start_time ,prediction , valeurV, lisV
    valeurV =[]
    lisV = []
    prediction = {
        'Focus': 0,
        'NoFocus': 0
    }
    vid = cv2.VideoCapture("http://192.168.1.13:8080/video")
    frame_count = 0
    start_time = time.time()
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # Détecter les visages
        face_boxes = detect_faces(frame)

        # Dessiner les cadres autour des visages
        for (x, y, w, h) in face_boxes:
            cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)  # Bleu (BGR)

        if recording:
            elapsed_time = time.time() - start_time
            if elapsed_time >= 15:
                current_folder_index = (current_folder_index + 1) % 4
                start_time = time.time()

            for (x, y, w, h) in face_boxes:
                face = frame[y:h, x:w]
                frame_filename = os.path.join(output_folders[current_folder_index], f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, face)
                frame_count += 1

        # Conversion de l'image en format JPEG
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route pour le flux vidéo
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route pour démarrer l'enregistrement des frames
@app.route('/start_record', methods=['POST'])
def start_record():
    global recording, start_time
    if not recording:
        start_time = time.time()
        recording = True
    return '', 204

# Route pour arrêter l'enregistrement des frames
@app.route('/stop_record', methods=['POST'])
def stop_record():
    global recording

    if recording:
        recording = False

        # hethy ll coupier mta tsawsr bel ligne space
        input_parent_folder = 'recordings'

        # Liste des sous-dossiers (folder_0, folder_1, folder_2, folder_3)
        subfolders = ['folder_0', 'folder_1', 'folder_2', 'folder_3']

        # Boucle sur chaque sous-dossier
        for subfolder in subfolders:
            input_folder = os.path.join(input_parent_folder, subfolder)
            output_folder = os.path.join('recordings', f'folderRet_{subfolder[-1]}')

            os.makedirs(output_folder, exist_ok=True)

            # Récupérer la liste des noms de fichiers d'images
            image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))]

            # Sélectionner 10 indices d'images à traiter
            selected_indices = np.linspace(0, len(image_files) - 1, 10, dtype=int)

            # Boucle sur chaque image sélectionnée
            for index in selected_indices:
                image_file = image_files[index]
                source_path = os.path.join(input_folder, image_file)
                destination_path = os.path.join(output_folder, image_file)

                # Copier l'image dans le dossier de destination
                shutil.copy2(source_path, destination_path)

        pred_vit()

    return '', 204


def load_video_images(video_path, H, W):
    images = []
    # Vérifier si le chemin fourni est un répertoire
    if os.path.isdir(video_path):
        for image_file in sorted(os.listdir(video_path)):
            image = tf.io.read_file(os.path.join(video_path, image_file))
            image = tf.image.decode_image(image, channels=3)
            image = tf.image.resize(image, (H, W))
            images.append(image)
    else:
        print(f"Le chemin fourni {video_path} n'est pas un répertoire.")
    return images


def create_video_dataset(train_dir, batch_size, num_frames, H, W):
    videos_data = []
    video_paths = []

    # Vérifier si train_dir est un répertoire
    if os.path.isdir(train_dir):
        video_images = load_video_images(train_dir, H, W)
        if len(video_images) >= num_frames:
            videos_data.append(video_images[:num_frames])
            video_paths.append(train_dir)
        print(f"Created dataset with {len(videos_data)} videos")  # Ajout d'une impression pour le débogage
    else:
        print(f"Le chemin train_dir {train_dir} n'est pas un répertoire.")

    if videos_data:
        videos_data = tf.convert_to_tensor(videos_data)
        dataset = tf.data.Dataset.from_tensor_slices((videos_data, video_paths))
        dataset = dataset.batch(batch_size)
        return dataset
    else:
        return None


def pred_vit():

    batch_size = 16
    num_frames = 9
    H, W = 96, 96


    video_paths = ["recordings/folderRet_0", "recordings/folderRet_1", "recordings/folderRet_2",
                   "recordings/folderRet_3"]

    for video_path in video_paths:
        dataset = create_video_dataset(video_path, batch_size, num_frames, H, W)

        if dataset:
            pred = model.predict(dataset)
            pred_classes = (pred < 0.5)
            prediction['Focus'] += np.sum(pred_classes == False)
            prediction['NoFocus'] += np.sum(pred_classes == True)
            valeurV.append(pred)
            lisV.append('Focus' if pred_classes == False else 'NoFocus')
        else:
            print(f"Aucune donnée valide trouvée dans {video_path}")

    print(prediction)
    print(valeurV)
    print(lisV)


def chart_values():
    return prediction

@app.route('/chart')
def chart():
    return jsonify({"Focus": int(prediction['Focus']),
                    "NoFocus": int(prediction['NoFocus'])})

@app.route('/res')
def res():
    return jsonify(lisV)

if __name__ == '__main__':
    app.run(debug=True)
