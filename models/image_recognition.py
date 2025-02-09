from transformers import pipeline

# Load pre-trained image classifier model
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# Load pre-trained text generation model
text_generator = pipeline("text-generation", model="gpt2")

def classify_image(image_path):
    """
    Classify an image and return predictions.
    """
    results = classifier(image_path)
    return results

def generate_description(label):
    """
    Generate a description for the identified object.
    """
    prompt = f"A {label} is commonly used for:"
    descriptions = text_generator(prompt, max_length=50, num_return_sequences=1)
    return descriptions[0]['generated_text']
