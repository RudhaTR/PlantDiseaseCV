from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
from PIL import Image




# Initialize Flask app
app = Flask(__name__)

class_labels = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


sorted(class_labels)


# Load the model
MODEL_PATH = "models\DiseaseDetectionCustommodel.h5"  # Update with your model path
model = load_model(MODEL_PATH)  # Load the model

UPLOAD_FOLDER = r"static\uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Define a route for the home page
# Routes
@app.route("/")
def index():
    return render_template("index.html")  # Ensure this matches your frontend template

@app.route('/disease')
def disease():
    return render_template('disease.html')  # Disease recognition page

@app.route('/cropRecommendation')
def cropRecommendation():
    return render_template('form.html')  # Disease recognition page

@app.route("/predict", methods=["POST"])
def predict():
    if "diseaseImage" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["diseaseImage"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded image
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        # Preprocess the image
        img = load_img(file_path, target_size=(224, 224))  # Ensure the size matches your model's input
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize if required by your model

        # Predict
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=-1)[0]
        #predicted_class = class_labels[np.argmax(predictions[0])]
        predicted_class = class_labels[predicted_class_index]
        confidence = np.max(predictions)

        # Return result
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.2f}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)