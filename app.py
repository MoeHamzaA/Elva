from flask import Flask, render_template, request, jsonify
from models.image_recognition import ImageClassifier
from models.flan_t5_usage import generate_usage_instructions
from models.detr_object_detection import detect_objects
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import base64
import os
import json
from gtts import gTTS
import tempfile
import uuid
from datetime import datetime
import glob
import shutil
import threading
import time
import pyttsx3

# Initialize InsightFace with Buffalo_L
face_app = FaceAnalysis(name="buffalo_l", root="./models")
face_app.prepare(ctx_id=-1)

# Delete the zip file if it exists
zip_files = glob.glob("./models/models/buffalo_l.zip")
for zip_file in zip_files:
    try:
        os.remove(zip_file)
        print(f"Deleted {zip_file}")
    except Exception as e:
        print(f"Error deleting zip file: {e}")

# Define database file
DATABASE_FILE = "face_database.json"

# Initialize the image classifier
classifier = ImageClassifier()

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Slower speech rate for clarity

app = Flask(__name__)

# Add configuration for accessibility
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['TEMP_FOLDER'] = 'static/temp'
app.config['MAX_TEMP_FILES'] = 50  # Maximum number of temporary files to keep
app.config['TEMP_FILE_AGE'] = 3600  # Maximum age of temporary files in seconds (1 hour)

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

def cleanup_temp_files():
    """Clean up old temporary files"""
    try:
        temp_dir = app.config['TEMP_FOLDER']
        current_time = time.time()
        
        # Get all files in temp directory
        for filename in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, filename)
            # Check if file is older than MAX_TEMP_FILE_AGE
            if os.path.getmtime(filepath) < current_time - app.config['TEMP_FILE_AGE']:
                try:
                    os.remove(filepath)
                    print(f"Cleaned up old temp file: {filename}")
                except Exception as e:
                    print(f"Error cleaning up {filename}: {e}")
        
        # If there are too many files, remove oldest ones
        files = os.listdir(temp_dir)
        if len(files) > app.config['MAX_TEMP_FILES']:
            files.sort(key=lambda x: os.path.getmtime(os.path.join(temp_dir, x)))
            for filename in files[:-app.config['MAX_TEMP_FILES']]:
                try:
                    os.remove(os.path.join(temp_dir, filename))
                    print(f"Removed excess temp file: {filename}")
                except Exception as e:
                    print(f"Error removing excess file {filename}: {e}")
    except Exception as e:
        print(f"Error in cleanup_temp_files: {e}")

def generate_voice_feedback(text):
    """Generate voice feedback for the given text."""
    try:
        # Create a unique filename for this feedback
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"static/voice_feedback/feedback_{timestamp}.mp3"
        
        # Ensure the directory exists
        os.makedirs("static/voice_feedback", exist_ok=True)
        
        # Generate the audio file
        engine.save_to_file(text, filename)
        engine.runAndWait()
        
        return filename
    except Exception as e:
        print(f"Error generating voice feedback: {str(e)}")
        return None

def load_database():
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, "r") as file:
            return json.load(file)
    return {}

def save_database(database):
    with open(DATABASE_FILE, "w") as file:
        json.dump(database, file)

def is_same_person(embedding1, embedding2, threshold=0.5):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity > threshold

def get_person_details(name, database):
    """Get detailed information about a person"""
    if name in database:
        data = database[name]
        return {
            "name": name,
            "description": data.get("description", ""),
            "added_date": data.get("added_date", ""),
            "last_seen": data.get("last_seen", ""),
            "relationship": data.get("relationship", ""),
            "notes": data.get("notes", "")
        }
    return None

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    try:
        data = request.json
        image_data = data['image']
        image_data = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        faces = face_app.get(frame)
        if not faces:
            message = "No faces detected. Please try again."
            voice_files = generate_voice_feedback(message)
            return jsonify({
                'message': message,
                'voice_feedback': voice_files
            })

        database = load_database()
        recognized_people = []
        person_details = []
        
        for face in faces:
            embedding = face.embedding.tolist()
            found = False
            for name, data in database.items():
                if is_same_person(embedding, data["embedding"]):
                    recognized_people.append(name)
                    details = get_person_details(name, database)
                    if details:
                        person_details.append(details)
                    found = True
                    # Update last seen
                    data["last_seen"] = str(datetime.now())
                    break
            if not found:
                recognized_people.append("Unknown")

        if recognized_people:
            if len(recognized_people) == 1:
                if recognized_people[0] == "Unknown":
                    message = "I don't recognize this person. Would you like me to remember them?"
                else:
                    person = person_details[0]
                    message = f"This is {person['name']}. {person['description']}"
                    if person.get('relationship'):
                        message += f" They are your {person['relationship']}."
            else:
                message = "I see multiple people: " + ", ".join(recognized_people)
                for person in person_details:
                    message += f"\n{person['name']} is {person['description']}"
            
            voice_files = generate_voice_feedback(message)
            return jsonify({
                'message': message,
                'voice_feedback': voice_files,
                'recognized_people': recognized_people,
                'person_details': person_details
            })

        message = "I don't recognize anyone in this photo. Would you like me to remember them?"
        voice_files = generate_voice_feedback(message)
        return jsonify({
            'message': message,
            'voice_feedback': voice_files
        })
    except Exception as e:
        message = "I'm having trouble processing your image. Please try again."
        voice_files = generate_voice_feedback(message)
        return jsonify({
            'message': message,
            'voice_feedback': voice_files,
            'error': str(e)
        }), 500

@app.route('/add_face', methods=['POST'])
def add_face():
    try:
        data = request.json
        name = data['name']
        description = data.get('description', '')
        relationship = data.get('relationship', '')
        notes = data.get('notes', '')
        image_data = data['image']
        image_data = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        faces = face_app.get(frame)
        if not faces:
            message = "No face detected. Please try again."
            voice_files = generate_voice_feedback(message)
            return jsonify({
                'message': message,
                'voice_feedback': voice_files
            })

        embedding = faces[0].embedding.tolist()
        database = load_database()
        database[name] = {
            "embedding": embedding,
            "description": description,
            "relationship": relationship,
            "notes": notes,
            "added_date": str(datetime.now()),
            "last_seen": str(datetime.now())
        }
        save_database(database)

        message = f"Great! I've saved {name}'s face. {description}"
        if relationship:
            message += f" They are your {relationship}."
        voice_files = generate_voice_feedback(message)
        return jsonify({
            'message': message,
            'voice_feedback': voice_files
        })
    except Exception as e:
        message = "I'm having trouble saving the face. Please try again."
        voice_files = generate_voice_feedback(message)
        return jsonify({
            'message': message,
            'voice_feedback': voice_files,
            'error': str(e)
        }), 500

@app.route('/get_remembered_people', methods=['GET'])
def get_remembered_people():
    try:
        database = load_database()
        people = []
        for name, data in database.items():
            people.append({
                "name": name,
                "description": data.get("description", ""),
                "relationship": data.get("relationship", ""),
                "notes": data.get("notes", ""),
                "added_date": data.get("added_date", ""),
                "last_seen": data.get("last_seen", "")
            })
        return jsonify({"people": people})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/update_person', methods=['POST'])
def update_person():
    try:
        data = request.json
        name = data['name']
        updates = data.get('updates', {})
        
        database = load_database()
        if name in database:
            database[name].update(updates)
            save_database(database)
            return jsonify({"message": f"Updated information for {name}"})
        return jsonify({"error": "Person not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_person', methods=['POST'])
def delete_person():
    try:
        data = request.json
        name = data['name']
        
        database = load_database()
        if name in database:
            del database[name]
            save_database(database)
            return jsonify({"message": f"Removed {name} from memory"})
        return jsonify({"error": "Person not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return render_template("index.html", title="Elva - Your Memory Assistant")

@app.route('/camera_detect')
def camera_detect():
    return render_template('camera_detect.html')

@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    try:
        data = request.json
        image_data = data['image']
        image_data = base64.b64decode(image_data.split(',')[1])

        upload_folder = app.config['UPLOAD_FOLDER']
        image_path = os.path.join(upload_folder, f"captured_image_{uuid.uuid4()}.jpg")
        with open(image_path, "wb") as f:
            f.write(image_data)

        return jsonify({"image_path": image_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    """Handle image classification requests."""
    try:
        if request.method == 'GET':
            # Handle GET request with image_path parameter
            image_path = request.args.get('image_path')
            if not image_path:
                return jsonify({'error': 'No image path provided'}), 400
        else:
            # Handle POST request with file upload
            if 'image' not in request.files:
                return jsonify({'error': 'No image provided'}), 400
                
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
                
            # Save the uploaded image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"static/uploads/image_{timestamp}.jpg"
            file.save(image_path)
        
        # Classify the image
        detected_objects = classifier.classify_image(image_path)
        
        # Generate descriptions and usage instructions
        description = classifier.generate_description(detected_objects)
        usage_instructions = classifier.get_usage_instructions(detected_objects)
        
        # Generate voice feedback
        voice_feedback = []
        
        # Add description to voice feedback
        desc_audio = generate_voice_feedback(description)
        if desc_audio:
            voice_feedback.append(desc_audio)
        
        # Add usage instructions to voice feedback
        usage_audio = generate_voice_feedback(usage_instructions)
        if usage_audio:
            voice_feedback.append(usage_audio)
        
        # Save detection history
        classifier.save_detection_history(image_path, detected_objects, description)
        
        return render_template('results.html',
                             image_path=image_path,
                             detected_objects=detected_objects,
                             usage_instructions=usage_instructions,
                             voice_feedback=voice_feedback)
                             
    except Exception as e:
        print(f"Error in classify route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route("/camera")
def camera():
    return render_template("camera.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/error")
def error():
    return render_template("error.html", message="An error occurred")

@app.route('/get_memory_timeline', methods=['GET'])
def get_memory_timeline():
    try:
        database = load_database()
        timeline = []
        for name, data in database.items():
            timeline.append({
                "name": name,
                "relationship": data.get("relationship", ""),
                "last_seen": data.get("last_seen", ""),
                "description": data.get("description", ""),
                "notes": data.get("notes", "")
            })
        # Sort by last seen date
        timeline.sort(key=lambda x: x["last_seen"], reverse=True)
        return jsonify({"timeline": timeline})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/add_memory_note', methods=['POST'])
def add_memory_note():
    try:
        data = request.json
        name = data['name']
        note = data['note']
        
        database = load_database()
        if name in database:
            if 'memory_notes' not in database[name]:
                database[name]['memory_notes'] = []
            database[name]['memory_notes'].append({
                'note': note,
                'timestamp': str(datetime.now())
            })
            save_database(database)
            return jsonify({"message": "Note added successfully"})
        return jsonify({"error": "Person not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_memory_game_data', methods=['GET'])
def get_memory_game_data():
    try:
        database = load_database()
        game_data = []
        for name, data in database.items():
            game_data.append({
                "name": name,
                "relationship": data.get("relationship", ""),
                "description": data.get("description", "")
            })
        return jsonify({"game_data": game_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/export_memories', methods=['GET'])
def export_memories():
    try:
        database = load_database()
        export_data = {
            "export_date": str(datetime.now()),
            "memories": []
        }
        
        for name, data in database.items():
            export_data["memories"].append({
                "name": name,
                "relationship": data.get("relationship", ""),
                "description": data.get("description", ""),
                "notes": data.get("notes", ""),
                "added_date": data.get("added_date", ""),
                "last_seen": data.get("last_seen", ""),
                "memory_notes": data.get("memory_notes", [])
            })
        
        return jsonify(export_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add context processor for accessibility controls
@app.context_processor
def inject_accessibility():
    return {
        'include_accessibility': True,
        'voice_feedback': []  # Default empty list for voice feedback
    }

if __name__ == "__main__":
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_temp_files, daemon=True)
    cleanup_thread.start()
    
    # Use production server if PORT environment variable is set (e.g., on Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
