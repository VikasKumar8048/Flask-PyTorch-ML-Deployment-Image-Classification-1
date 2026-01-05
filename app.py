from flask import Flask, request, render_template
from PIL import Image
import io

from network import classify_image

app = Flask(__name__, template_folder="templates")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    # Check if file exists
    if "image" not in request.files:
        return "❌ No file part in request"

    file = request.files["image"]

    if file.filename == "":
        return "❌ No file selected"

    try:
        # Read image
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert("RGB")  # VERY IMPORTANT for CNNs

        # Predict
        prediction = classify_image(img)

        return f"✅ Classification: {prediction}"

    except Exception as e:
        return f"❌ Error: {str(e)}"


# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

