from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image

# Load DETR model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

def detect_objects(image_path):
    """
    Perform object detection using the DETR model.
    :param image_path: Path to the input image
    :return: List of detected objects with labels, confidence scores, and bounding boxes
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Get predictions from the model
    outputs = model(**inputs)

    # Post-process predictions (threshold = 0.9 for high confidence)
    target_sizes = torch.tensor([image.size[::-1]])  # Target image size
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    
    # Define labels to exclude (human-related)
    exclude_labels = {"person", "hand", "foot", "arm", "leg", "face"}  # Add more if needed

    # Extract detected objects
    # Extract detected objects and filter out unwanted labels
    detected_objects = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label.item()]
        if label_name not in exclude_labels:
            box = [round(i, 2) for i in box.tolist()]  # Round bounding box coordinates
            detected_objects.append({
                "label": label_name,
                "score": round(score.item(), 3),
                "box": box
            })

    return detected_objects
