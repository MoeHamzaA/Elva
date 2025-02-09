# Alzheimer's Aid App, Elva

The Alzheimer's Aid App, Known as Elva, is an AI-powered web application designed to assist individuals with Alzheimer's and dementia in identifying everyday objects and recognizing faces. The app incorporates live camera integration, object detection, and facial recognition to simplify day-to-day tasks and promote independence.

Features

Live Camera Integration: Users can capture images directly using a live camera feed.

Object Detection: AI-based object detection helps users identify objects in their surroundings.

Facial Recognition: Recognizes and identifies faces using a stored database of embeddings.

Custom Usage Instructions: Provides step-by-step guidance on how to use the detected objects.

User-Friendly Interface: Simple and consistent UI design for ease of use.

Tech Stack

Backend: Python, Flask

Frontend: HTML, CSS, JavaScript

AI Models:

InsightFace for facial recognition

DETR for object detection

Database: JSON-based storage for face embeddings

Project Structure

.
├── app.py                  # Main Flask application
├── static/
│   ├── css/                # CSS files for styling
│   ├── js/
│   │   └── camera.js       # JavaScript for camera functionality
│   └── uploads/            # Directory for uploaded images
├── templates/
│   ├── index.html          # Homepage
│   ├── detect.html         # Upload or take a photo for object detection
│   ├── results.html        # Display object detection results
│   ├── camera.html         # Live camera feed for facial recognition
│   └── camera_detect.html  # Live camera feed for object detection
├── face_database.json      # JSON database for facial recognition
└── README.md               # Project documentation

Setup and Installation

Prerequisites

Python 3.8 or later

Flask

OpenCV

InsightFace

DETR and related dependencies

Installation

Clone the repository:

git clone https://github.com/your-username/alzheimers-aid-app.git
cd alzheimers-aid-app

Set up a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Download required AI models:

Place InsightFace models in the models/ directory.

Set up DETR models for object detection.

Run the app:

python app.py

Open the app in your browser:

http://127.0.0.1:5000

How to Use

Homepage:

Navigate to the homepage to choose between object detection or facial recognition.

Object Detection:

Upload an image or use the live camera to capture a photo.

View the detected objects along with step-by-step instructions on how to use them.

Facial Recognition:

Use the live camera to recognize faces or add new faces to the database.

Results Page:

Review the detection results and return to the homepage for further actions.

Future Enhancements

Add multilingual support for a wider user base.

Implement user accounts for personalized face and object databases.

Integrate audio descriptions for users with visual impairments.

Expand the object detection model to cover more categories.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

InsightFace: For providing a robust facial recognition library.

DETR: For enabling efficient object detection.

Flask: For serving as the backbone of the web application.

Special thanks to all contributors and testers who made this project possible.
