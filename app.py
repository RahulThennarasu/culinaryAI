# app.py

from flask import Flask, render_template, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline, set_seed

# Initialize Flask application
app = Flask(__name__)

# Load fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./fine-tuned-gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('./fine-tuned-gpt2')

# Create pipeline
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=-1)  # Ensure the CPU is used

# Set the seed for reproducibility
set_seed(42)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle recipe generation
@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    try:
        prompt = request.json['prompt']
        generated = generator(prompt, max_length=1000, num_return_sequences=1, top_p=0.9)
        return jsonify({'generated_text': generated[0]['generated_text']})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
