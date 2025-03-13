from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_response(model_id: str, prompt: str):
    """Load the model and tokenizer from the Hugging Face Hub and generate a response."""
    
    # Load the tokenizer and model from the Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).cuda()
    
    # Tokenize the prompt and move input tensors to GPU
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")
    
    # Generate a response
    output = model.generate(
        inputs["input_ids"],
        max_length=256,  # Maximum length of the response
        num_beams=5,     # Number of beams for beam search
        early_stopping=True,
        no_repeat_ngram_size=2,  # Avoid repetition
        temperature=0.7,         # Control the randomness of predictions
    )
    
    # Decode and return the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Set model ID and prompt as variables
model_id = "Vipas-AI/llama-retrained-disease-dataset"  # Example model ID, replace with the actual one
prompt = "What is the future of artificial intelligence?"

# Generate response
response = generate_response(model_id, prompt)

# Print results
print("Prompt:", prompt)
print("Response:", response)
