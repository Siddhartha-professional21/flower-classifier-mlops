# ============================================================
# app.py - FastAPI Web Server for Flower Classifier
# ============================================================
# What this file does:
# 1. Loads your trained flower model (flower_classifier.h5)
# 2. Creates a web API server using FastAPI
# 3. Accepts an image file from the user
# 4. Resizes and prepares the image
# 5. Runs the model and returns the predicted flower name
# 6. Also has a simple homepage to test it works
# ============================================================


# ---- Import all required libraries ----
from fastapi import FastAPI, UploadFile, File   # FastAPI = web framework
from fastapi.responses import HTMLResponse       # to return HTML pages
import tensorflow as tf                          # to load our trained model
import numpy as np                               # for number operations
from PIL import Image                            # to open and resize images
import io                                        # to read uploaded file bytes
import uvicorn                                   # to run the server


# ============================================================
# STEP 1: Create the FastAPI app
# ============================================================
# FastAPI is like a waiter in a restaurant
# - You (client) send a request (order)
# - FastAPI processes it and returns a response (food)

app = FastAPI(
    title="🌸 Flower Classifier API",
    description="Upload a flower image and get the predicted flower name!",
    version="1.0"
)


# ============================================================
# STEP 2: Load the trained model ONCE when server starts
# ============================================================
# We load the model only ONCE at startup (not every request)
# This makes predictions much faster!

print("Loading flower classifier model...")
model = tf.keras.models.load_model("model/flower_classifier.keras")
print("Model loaded successfully!")

# These are our 5 flower class names (must match training order = alphabetical)
CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Image size must match what we used in training
IMAGE_SIZE = (96, 96)


# ============================================================
# STEP 3: Homepage Route - just to confirm server is running
# ============================================================
# When you open http://127.0.0.1:8000 in browser, you see this page

@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_content = """
    <html>
        <head>
            <title>Flower Classifier</title>
            <style>
                body { font-family: Arial; text-align: center; padding: 50px; background: #f0f8f0; }
                h1   { color: #2e7d32; }
                p    { color: #555; font-size: 18px; }
                a    { color: #2196F3; font-size: 18px; }
                .box { background: white; padding: 30px; border-radius: 10px;
                       display: inline-block; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            </style>
        </head>
        <body>
            <div class="box">
                <h1>🌸 Flower Classifier API</h1>
                <p>Your flower classification model is <b>running!</b></p>
                <p>Can classify: 🌼 Daisy | 🌻 Dandelion | 🌹 Rose | 🌻 Sunflower | 🌷 Tulip</p>
                <br>
                <a href="/docs">👉 Click here to test the API (Swagger UI)</a>
                <br><br>
                <a href="/health">👉 Check server health</a>
            </div>
        </body>
    </html>
    """
    return html_content


# ============================================================
# STEP 4: Health Check Route
# ============================================================
# Useful to confirm the server and model are working fine

@app.get("/health")
async def health_check():
    return {
        "status"     : "✅ Server is running",
        "model"      : "✅ Model is loaded",
        "classes"    : CLASS_NAMES,
        "image_size" : IMAGE_SIZE
    }


# ============================================================
# STEP 5: MAIN PREDICTION ROUTE - The most important part!
# ============================================================
# This route accepts an image file and returns the prediction
# Method: POST (because we are SENDING data to the server)
# URL: http://127.0.0.1:8000/predict

@app.post("/predict")
async def predict(file: UploadFile = File(...)):   # File(...) means file is required
    """
    Upload a flower image and get back:
    - predicted flower name
    - confidence score (how sure the model is)
    - probabilities for all 5 flowers
    """

    # ---- Check if uploaded file is an image ----
    # We only accept jpg, jpeg, and png files
    allowed_types = ["image/jpeg", "image/jpg", "image/png"]

    if file.content_type not in allowed_types:
        return {
            "error": f"Please upload a JPG or PNG image. You uploaded: {file.content_type}"
        }

    # ---- Read the uploaded image file ----
    # file.read() gives us the raw bytes of the image
    image_bytes = await file.read()

    # Convert bytes → PIL Image object (so we can resize it)
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB (in case someone uploads a PNG with transparency = 4 channels)
    # Our model expects 3 channels (Red, Green, Blue) only
    image = image.convert("RGB")

    # ---- Resize image to 96x96 (same size used in training) ----
    image = image.resize(IMAGE_SIZE)

    # ---- Convert image to numpy array ----
    # Model expects numbers, not image objects
    img_array = np.array(image)            # shape: (96, 96, 3)

    # ---- Normalize pixel values from 0-255 → 0-1 ----
    # Same normalization we did in training (rescale=1/255)
    img_array = img_array / 255.0          # shape: (96, 96, 3)

    # ---- Add batch dimension ----
    # Model expects shape: (batch_size, height, width, channels)
    # We have 1 image, so: (1, 96, 96, 3)
    img_array = np.expand_dims(img_array, axis=0)   # shape: (1, 96, 96, 3)

    # ---- Run the model prediction ----
    predictions = model.predict(img_array)   # returns array of 5 probabilities
    # Example output: [[0.02, 0.05, 0.85, 0.03, 0.05]]
    # This means: daisy=2%, dandelion=5%, rose=85%, sunflower=3%, tulip=5%

    # ---- Get the predicted class ----
    predicted_index = int(np.argmax(predictions[0]))   # index of highest probability
    predicted_class = CLASS_NAMES[predicted_index]      # name of the flower
    confidence      = float(np.max(predictions[0]))     # highest probability value

    # ---- Build probabilities dictionary for all 5 classes ----
    all_probabilities = {
        CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)   # convert to percentage
        for i in range(len(CLASS_NAMES))
    }

    # ---- Return the result as JSON ----
    return {
        "filename"        : file.filename,
        "predicted_flower": predicted_class,
        "confidence"      : f"{confidence * 100:.2f}%",
        "all_probabilities": all_probabilities,
        "message"         : f"This flower is most likely a {predicted_class.upper()}! 🌸"
    }


# ============================================================
# STEP 6: Run the server (only when running this file directly)
# ============================================================
# This block runs ONLY when you type: python app.py
# It does NOT run when imported by another file

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  🌸 Starting Flower Classifier API Server")
    print("="*50)
    print("  Open in browser: http://127.0.0.1:8000")
    print("  API docs page  : http://127.0.0.1:8000/docs")
    print("  Health check   : http://127.0.0.1:8000/health")
    print("  To STOP server : Press Ctrl + C")
    print("="*50 + "\n")

    uvicorn.run(
        "app:app",          # "filename:FastAPI_variable_name"
        host="0.0.0.0",     # accessible from any device on network
        port=8000,          # port number
        reload=False        # set True if you want auto-reload on code changes
    )