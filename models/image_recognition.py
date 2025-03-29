import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn.functional as F
import json
import os
from datetime import datetime
from transformers import DetrImageProcessor, DetrForObjectDetection

class ImageClassifier:
    def __init__(self):
        # Load DETR model and processor
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.model.eval()
        
        # Load common object descriptions
        with open('models/object_descriptions.json', 'r') as f:
            self.object_descriptions = json.load(f)

    def classify_image(self, image_path):
        """Classify the image and return detected objects with confidence scores and bounding boxes."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process predictions (threshold = 0.7 for good balance)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes, 
                threshold=0.7
            )[0]
            
            # Extract detected objects
            detected_objects = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                label_name = self.model.config.id2label[label.item()]
                confidence = score.item()
                box = [round(i, 2) for i in box.tolist()]
                
                print(f"Detected: {label_name} with confidence: {confidence:.2f}")  # Debug output
                
                detected_objects.append({
                    'label': label_name,
                    'score': confidence,
                    'box': box
                })
            
            return detected_objects
            
        except Exception as e:
            print(f"Error in classify_image: {str(e)}")
            return []

    def generate_description(self, detected_objects):
        """Generate a helpful description of the detected objects."""
        if not detected_objects:
            return "I'm not sure what I'm looking at. Could you try taking another photo?"
        
        descriptions = []
        for obj in detected_objects:
            label = obj['label'].lower()
            confidence = obj['score']
            
            # Get specific description if available
            description = self.object_descriptions.get(label, "")
            
            if confidence > 0.8:
                if description:
                    descriptions.append(f"I'm very confident this is a {label}. {description}")
                else:
                    descriptions.append(f"I'm very confident this is a {label}.")
            elif confidence > 0.6:
                if description:
                    descriptions.append(f"This looks like a {label}. {description}")
                else:
                    descriptions.append(f"This looks like a {label}.")
            else:
                if description:
                    descriptions.append(f"This might be a {label}. {description}")
                else:
                    descriptions.append(f"This might be a {label}.")
        
        # Combine descriptions
        if len(descriptions) == 1:
            return descriptions[0]
        else:
            return " ".join(descriptions)

    def get_usage_instructions(self, detected_objects):
        """Generate usage instructions based on detected objects."""
        if not detected_objects:
            return "I'm not sure how to help with this object. Could you try taking another photo?"
        
        instructions = []
        for obj in detected_objects:
            label = obj['label'].lower()
            
            # Get usage instructions if available
            usage = self.object_descriptions.get(label, {}).get('usage', "")
            if usage:
                instructions.append(usage)
        
        if instructions:
            return " ".join(instructions)
        else:
            return "I can see this object, but I don't have specific instructions for using it. Would you like me to explain what I see?"

    def save_detection_history(self, image_path, detected_objects, description):
        """Save detection history for future reference."""
        try:
            history_file = 'models/detection_history.json'
            history = []
            
            # Load existing history
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
            
            # Add new detection
            history.append({
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'detected_objects': detected_objects,
                'description': description
            })
            
            # Keep only last 100 detections
            if len(history) > 100:
                history = history[-100:]
            
            # Save updated history
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            print(f"Error saving detection history: {str(e)}")
