from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from io import BytesIO
import numpy as np
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)


# Load the pre-trained model
model = tf.keras.models.load_model(r"C:\Users\rufus\Documents\lung_disease_model.keras")

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY is not set. Ensure it's defined in your .env file.")
else:
    print("Loaded API Key successfully.")

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')  # About page

@app.route('/contact')
def contact():
    return render_template('contact.html')  # Contact page

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Convert the uploaded file to a BytesIO object and open it as an image
        img = Image.open(BytesIO(file.read()))
        img = img.resize((150, 150))  # Resize to the target size used in training
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize to match the model’s training scale

        # Make a prediction
        prediction = model.predict(img_array)
        prediction_class = np.argmax(prediction, axis=1)[0]

        # Define class names based on your model's output
        classes = ["Bacterial Pneumonia", "Corona Virus Disease", "Normal", "Tuberculosis", "Viral Pneumonia"]
        result = classes[prediction_class]

        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': 'Error processing the image', 'details': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'Message is required'}), 400

    # Prepare the payload
    payload = {
        "contents": [
            {
                "parts": [{"text": user_message}]
            }
        ]
    }

    # Include the API key in the URL (mimic the curl request)
    api_url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    # Set up headers
    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Make the request to the Gemini API
        response = requests.post(api_url_with_key, headers=headers, json=payload)

        if response.status_code == 200:
            gemini_response = response.json()
            raw_reply = gemini_response["candidates"][0]["content"]["parts"][0]["text"]

            # Format the response (e.g., replace newlines with <br> and add bullet points)
            cleaned_reply = (
                raw_reply
                .replace("••", "")          # Remove unwanted symbols
                .replace("*", "&bull;")    # Replace asterisks with bullet points
                .replace("\n", "<br>")     # Replace newlines with HTML line breaks
                .strip()                   # Remove extra spaces
            )

            return jsonify({'response': cleaned_reply})
        else:
            return jsonify({
                'error': 'Failed to communicate with Gemini API',
                'details': response.text
            }), response.status_code
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
