from flask import Flask, render_template, request, jsonify
from models.image_recognition import classify_image, generate_description
from models.flan_t5_usage import generate_usage_instructions
from models.detr_object_detection import detect_objects


import cv2
import numpy as np
from insightface.app import FaceAnalysis
import base64
import os
import json

# Initialize InsightFace with Buffalo_L
face_app = FaceAnalysis(name="buffalo_l", root="./models")  # Ensure the models folder exists
face_app.prepare(ctx_id=-1)  # Use CPU (ctx_id=-1)


# Define database file
DATABASE_FILE = "face_database.json"

app = Flask(__name__)

def load_database():
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, "r") as file:
            return json.load(file)
    return {}

def save_database(database):
    with open(DATABASE_FILE, "w") as file:
        json.dump(database, file)

# Compare embeddings (cosine similarity)
def is_same_person(embedding1, embedding2, threshold=0.5):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity > threshold



@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    data = request.json
    image_data = data['image']
    image_data = base64.b64decode(image_data.split(',')[1])  # Decode base64
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Convert to OpenCV image

    # Perform face recognition
    faces = face_app.get(frame)
    if not faces:
        return jsonify({'message': 'No faces detected'})

    database = load_database()
    for face in faces:
        embedding = face.embedding.tolist()

        # Check if the face is in the database
        for name, data in database.items():
            if is_same_person(embedding, data["embedding"]):
                return jsonify({'message': f'Hello, {name}!'})

    return jsonify({'message': 'Face not recognized. Please add to the database.'})

@app.route('/add_face', methods=['POST'])
def add_face():
    data = request.json
    name = data['name']
    image_data = data['image']
    image_data = base64.b64decode(image_data.split(',')[1])  # Decode base64
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Convert to OpenCV image

    # Perform face detection
    faces = face_app.get(frame)
    if not faces:
        return jsonify({'message': 'No face detected. Please try again.'})

    # Save the first detected face embedding
    embedding = faces[0].embedding.tolist()
    database = load_database()
    database[name] = {"embedding": embedding}
    save_database(database)

    return jsonify({'message': f'{name} has been added to the database.'})


@app.route("/")
def home():
    return render_template("index.html", title="Alzheimer's Aid App")

@app.route('/camera_detect')
def camera_detect():
    return render_template('camera_detect.html')


@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    data = request.json
    image_data = data['image']
    image_data = base64.b64decode(image_data.split(',')[1])  # Decode base64

    # Save the image to the uploads folder
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)  # Create folder if it doesn't exist
    image_path = os.path.join(upload_folder, "captured_image.jpg")
    with open(image_path, "wb") as f:
        f.write(image_data)

    # Return the image path to the frontend
    return jsonify({"image_path": image_path})

@app.route("/classify", methods=["GET", "POST"])
def classify():
     # Determine the image path
    if request.method == 'POST':
        # If an image is uploaded
        image = request.files["image"]
        image_path = f"static/uploads/{image.filename}"
        image.save(image_path)
    elif request.method == 'GET':
        # If the image path is passed via query string
        image_path = request.args.get('image_path')

    # Detect objects using DETR
    detected_objects = detect_objects(image_path)

    # Use the highest-confidence object for further processing
    if detected_objects:
        highest_confidence_object = max(detected_objects, key=lambda x: x["score"])
        highest_label = highest_confidence_object["label"]
        usage_instructions = generate_usage_instructions(highest_label)
    else:
        highest_label = "Unknown"
        usage_instructions = "Could not identify the object."

    # Render the results page
    return render_template(
        "results.html",
        image_path=image_path,
        detected_objects=detected_objects,
        highest_label=highest_label,
        usage_instructions=usage_instructions
    )
    
@app.route("/camera")
def camera():
    return render_template("camera.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

if __name__ == "__main__":
    app.run(debug=True)
