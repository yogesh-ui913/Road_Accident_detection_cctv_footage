from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model(
    "model_vgg16_aug_fine_tune.keras"
)

IMG_SIZE = (224, 224)


@app.route("/")
def home():
    return "Road Accident Detection API is running!"


@app.route("/predict", methods=["POST"])
def predict():

    try:
        if "file" not in request.files:
            return jsonify({
                "error": "No image file provided"
            }), 400

        file = request.files["file"]

        # Open image
        img = Image.open(
            io.BytesIO(file.read())
        ).convert("RGB")

        # Resize
        img = img.resize(IMG_SIZE)

        # Convert to array
        img_array = np.array(img)

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # Same preprocessing as training
        img_array = img_array / 255.0

        # Prediction
        prediction = model.predict(
            img_array,
            verbose=0
        )

        probability = float(
            prediction[0][0]
        )

        # 0 = Accident
        # 1 = Non Accident
        if probability >= 0.5:

            predicted_class = "Non Accident"
            confidence = probability

        else:

            predicted_class = "Accident"
            confidence = 1 - probability

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
