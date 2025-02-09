from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load FLAN-T5 Large model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

def remove_duplicates(text):
    """
    Remove duplicate sentences from the text.
    :param text: The input text with possible repetitions.
    :return: Text with duplicates removed.
    """
    sentences = text.split('. ')
    seen = set()
    result = []
    for sentence in sentences:
        if sentence not in seen:
            seen.add(sentence)
            result.append(sentence)
    return '. '.join(result).strip()

def generate_usage_instructions(object_name, max_length=150, num_beams=5, early_stopping=True):
    """
    Generate step-by-step usage instructions using FLAN-T5 Large.
    :param object_name: Name of the object
    :param max_length: Maximum length of the generated response (default: 150).
    :param num_beams: Number of beams for beam search (default: 5).
    :param early_stopping: Whether to stop early during beam search (default: True).
    :return: Generated instructions as a string with duplicates removed.
    """
    if not object_name or not isinstance(object_name, str):
        return "Error: Please provide a valid object name as a string."

    # Define the prompt with a dynamic example
    prompt = (
        f"Provide clear, practical, and step-by-step instructions on how to use a {object_name}. "
        f"The instructions should be easy to follow and suitable for everyday use. "
        f"For example, if the object is 'toothbrush,' the steps might include: "
        f"1. Apply toothpaste to the bristles. 2. Hold the toothbrush at a 45-degree angle. 3. Brush gently in circular motions. "
        f"Now, explain how to use a {object_name} in a similar, step-by-step manner."
    )

    try:
        # Tokenize the input prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        # Generate the response
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
        )

        # Decode and remove duplicates from the response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return remove_duplicates(generated_text)
    except Exception as e:
        return f"An error occurred: {str(e)}"
