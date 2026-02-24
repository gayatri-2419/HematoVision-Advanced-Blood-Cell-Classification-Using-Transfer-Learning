import os
import numpy as np
import cv2
import base64
from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load trained model
model = load_model("Blood_Cell.h5")

# Class labels
class_labels = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

# Prediction Function
def predict_image_class(image_path, model):
    img = cv2.imread(image_path)

    if img is None:
        return "Invalid Image", None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    img_preprocessed = preprocess_input(
        img_resized.reshape((1, 224, 224, 3))
    )
    
    predictions = model.predict(img_preprocessed)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_idx]
    
    return predicted_class_label, img_rgb


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        if file:
            if not os.path.exists("static"):
                os.makedirs("static")

            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            predicted_class_label, img_rgb = predict_image_class(file_path, model)

            if img_rgb is not None:
                success, buffer = cv2.imencode(
                    '.png',
                    cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                )

                if success:
                    img_str = base64.b64encode(buffer.tobytes()).decode('utf-8')
                else:
                    img_str = None
            else:
                img_str = None

            return render_template(
                "result.html",
                class_label=predicted_class_label,
                img_data=img_str
            )

    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)