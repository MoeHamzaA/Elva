# Elva – Alzheimer's Aid App 🧠✨  
### 🥉 3rd Place Winner at HackHive Hackathon 2025

**Elva** is an AI-powered web application designed to assist individuals with Alzheimer's and dementia in recognizing familiar faces and identifying everyday objects. With live camera integration, object detection, facial recognition, and usage guidance, Elva empowers users to live more independently.

---

## 🧩 Features

- 📸 **Live Camera Integration**: Capture images directly using a live camera feed.
- 🧠 **Object Detection**: AI-powered object identification using DETR.
- 🙂 **Facial Recognition**: Recognizes and identifies people from a stored facial database using InsightFace.
- 📝 **Custom Usage Instructions**: Displays step-by-step guidance on how to use detected objects.
- 🖥️ **User-Friendly Interface**: Simple, clean, and accessible UI for all users.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **AI Models**:
  - [InsightFace](https://github.com/deepinsight/insightface) for facial recognition
  - [DETR](https://github.com/facebookresearch/detectron2) for object detection
- **Database**: JSON-based storage for face embeddings

---

## 🚀 Setup & Installation

### 📋 Prerequisites

- Python 3.8+
- Flask
- OpenCV
- InsightFace
- DETR and its dependencies

### 📦 Installation Steps

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/alzheimers-aid-app.git
    cd alzheimers-aid-app
    ```

2. **Set up a virtual environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Download AI Models**:
    - Place **InsightFace** models in the `models/` directory.
    - Set up **DETR** models according to its documentation.

5. **Run the application**:

    ```bash
    python app.py
    ```

6. **Access in browser**:

    ```
    http://127.0.0.1:5000
    ```

---

## 🧪 How to Use

- **Homepage**: Choose between *Object Detection* or *Facial Recognition*.
- **Object Detection**: Upload or capture an image → View detected objects → Get usage instructions.
- **Facial Recognition**: Capture or upload a face → Match against database → Optionally add new faces.
- **Results Page**: Review results and navigate back for additional tasks.

---

## 🌱 Future Enhancements

- 🌍 Multilingual support
- 👤 Personalized user accounts
- 🔊 Audio descriptions for visually impaired users
- 🧾 Expanded object category support

---

## 🔗 Project Links

- 🔹 [Devpost Submission – MindGuide (Elva)](https://devpost.com/software/mindguide-6bms3h)

---

## 📜 License

This project is licensed under the [MIT License](./LICENSE).

---

## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface): Facial Recognition
- [DETR](https://github.com/facebookresearch/detectron2): Object Detection
- [Flask](https://flask.palletsprojects.com): Web Framework

Special thanks to our contributors and testers for their time and feedback 💙

---
