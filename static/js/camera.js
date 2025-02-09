const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('capture');
const resultText = document.getElementById('result');

// Access the user's webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(error => {
        console.error('Error accessing webcam:', error);
    });

// Freeze the camera
function freezeCamera() {
    const stream = video.srcObject;
    const tracks = stream.getTracks();
    tracks.forEach(track => track.stop()); // Stop the video stream
    video.srcObject = null;
}

// Capture the photo and send it to the backend
captureButton.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL('image/jpeg');

    // Freeze the camera
    freezeCamera();

    // Send the captured image to the Flask backend
    fetch('/recognize_face', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: imageData })
    })
        .then(response => response.json())
        .then(data => {
            resultText.textContent = data.message;

            // If face is not recognized, prompt for name
            if (data.message.includes('not recognized')) {
                const name = prompt('Face not recognized. Please enter your name:');
                if (name) {
                    // Send the name and image to the backend to add to the database
                    fetch('/add_face', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ name: name, image: imageData })
                    })
                        .then(response => response.json())
                        .then(data => {
                            resultText.textContent = data.message;
                        })
                        .catch(error => {
                            console.error('Error:', error);
                        });
                }
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
});