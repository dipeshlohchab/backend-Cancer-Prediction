from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
import json
import os
import traceback
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download # <--- NEW IMPORT: Added huggingface_hub

# --- Global Variables for Model, Class Names, and Descriptions ---
MODEL = None
CLASS_NAMES = []
CLASS_DESCRIPTIONS = {}

# --- Hugging Face Model Details ---
# These are derived from your provided Hugging Face repository URL
HF_REPO_ID = "dipeshlohchab0302/Cancer_Prediction"
HF_MODEL_FILENAME = "Cancer_Classification.keras" # Assuming this is the name of your model file in the Hugging Face repo

MODEL_PATH="./Cancer_Classification.keras" # This is the local path where the model will be downloaded
# --- Paths to your local JSON data files (these are still needed locally or in your repo) ---
CLASS_NAMES_PATH = "class_names.json"
CLASS_DESCRIPTIONS_PATH = "class_description.json"

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Downloads the Keras model from Hugging Face Hub with a local fallback.
    """
    global MODEL, CLASS_NAMES, CLASS_DESCRIPTIONS
    print("Application startup initiated. Loading resources...")

    # --- 1. Load Keras Model (Your existing logic is good) ---
    try:
        print(f"Attempting to download model from Hugging Face Hub: {HF_REPO_ID}/{HF_MODEL_FILENAME}...")
        downloaded_model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_FILENAME,
            local_dir='./',
            local_dir_use_symlinks=False
        )
        MODEL = tf.keras.models.load_model(downloaded_model_path)
        print("Keras model loaded successfully from Hugging Face Hub.")
    except Exception as hf_error:
        print(f"WARNING: Hugging Face download failed. Reason: {hf_error}")
        print(f"Attempting to load model from local fallback path: {MODEL_PATH}")
        try:
            MODEL = tf.keras.models.load_model(MODEL_PATH)
            print("Keras model loaded successfully from local path.")
        except Exception as local_error:
            raise RuntimeError(f"Critical: Model failed to download/load from both sources. Application cannot start. Error: {local_error}")

    if MODEL:
        MODEL.summary()

    # --- 2. Load Class Names (CRITICAL FIX) ---
    try:
        print(f"Attempting to load class names from {CLASS_NAMES_PATH}...")
        with open(CLASS_NAMES_PATH, 'r') as f:
            CLASS_NAMES = json.load(f)
        print(f"Class names loaded successfully. Total: {len(CLASS_NAMES)} classes.")
    except Exception as e:
        raise RuntimeError(f"Critical: Class names failed to load from {CLASS_NAMES_PATH}. Application cannot start. Error: {e}")

    # --- 3. Load Class Descriptions (MOVED TO CORRECT LOCATION) ---
    try:
        print(f"Attempting to load class descriptions from {CLASS_DESCRIPTIONS_PATH}...")
        with open(CLASS_DESCRIPTIONS_PATH, 'r') as f:
            CLASS_DESCRIPTIONS = json.load(f)
        print(f"Class descriptions loaded successfully. Total: {len(CLASS_DESCRIPTIONS)} entries.")
    except Exception as e:
        raise RuntimeError(f"Critical: Class descriptions failed to load. Application cannot start. Error: {e}")
        
    # --- 4. Consistency Checks ---
    if len(CLASS_NAMES) != MODEL.output_shape[1]:
        print(f"WARNING: Model expects {MODEL.output_shape[1]} classes, but {len(CLASS_NAMES)} class names were loaded.")
    
    missing_descriptions = [name for name in CLASS_NAMES if normalize_class_name_for_lookup(name) not in CLASS_DESCRIPTIONS]
    if missing_descriptions:
        print(f"WARNING: Missing descriptions for {len(missing_descriptions)} classes.")

    # --- 5. Application is ready ---
    print("Application startup complete. Server is ready.")
    yield # <--- All startup logic is BEFORE yield

    # --- Shutdown Logic ---
    print("Application shutdown initiated.")
    MODEL = None
    CLASS_NAMES = []
    CLASS_DESCRIPTIONS = {}
    print("Resources cleaned up. Application shutdown complete.")

# --- Initialize FastAPI Application (No change) ---
app = FastAPI(
    title="Medical Image Classification API",
    description="A FastAPI backend for classifying various medical images using a pre-trained Keras model. "
                "Provides human-readable predictions and descriptions.",
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS Configuration (Added 127.0.0.1:5500 as requested) ---
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5500", # <--- ADDED THIS ORIGIN
    # Add your Render frontend and backend URLs here after deployment
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Image Preprocessing Function (No change) ---
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """
    Preprocesses a PIL Image for a Keras model.
    Steps:
    1. Converts image to RGB (if not already).
    2. Resizes image to the target dimensions.
    3. Converts PIL Image to a NumPy array.
    4. Adds a batch dimension (e.g., (height, width, channels) -> (1, height, width, channels)).

    IMPORTANT: This function assumes that the Keras model itself contains a
    `tf.keras.layers.Rescaling(1./255)` layer or similar for pixel normalization.
    Therefore, no explicit 1/255.0 division is performed here.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target_size)
    image_array = np.asarray(image)

    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# --- Helper function for class name lookup (No change from previous) ---
def normalize_class_name_for_lookup(raw_name: str) -> str:
    if raw_name in CLASS_DESCRIPTIONS:
        return raw_name
    parts_underscore = raw_name.split('_')
    candidate_underscore = '_'.join([p.capitalize() for p in parts_underscore])
    if candidate_underscore in CLASS_DESCRIPTIONS:
        return candidate_underscore
    parts_space = raw_name.split(' ')
    candidate_space = ' '.join([p.capitalize() for p in parts_space])
    if candidate_space in CLASS_DESCRIPTIONS:
        return candidate_space
    candidate_title = raw_name.title()
    if candidate_title in CLASS_DESCRIPTIONS:
        return candidate_title
    candidate_first_cap = raw_name[0].upper() + raw_name[1:] if raw_name else ""
    if candidate_first_cap in CLASS_DESCRIPTIONS:
        return candidate_first_cap
    return raw_name

# --- Prediction Endpoint (No change) ---
@app.post("/predict",
          summary="Classify an uploaded image",
          response_description="Returns filename and top 3 predictions with details")
async def predict_image(file: UploadFile = File(..., description="An image file (JPG, PNG, etc.) to classify.")):
    if MODEL is None or not CLASS_NAMES or not CLASS_DESCRIPTIONS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not fully initialized. Model, class names, or descriptions were not loaded. Please check server logs for startup errors."
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image. Please upload an image file (e.g., JPG, PNG)."
        )

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        processed_image = preprocess_image(image, target_size=(224, 224))
        predictions = MODEL.predict(processed_image)

        top_n = 3
        top_indices = np.argsort(predictions[0])[::-1][:top_n]

        results = []
        for i in top_indices:
            raw_class_name = CLASS_NAMES[i]
            key_for_description_lookup = normalize_class_name_for_lookup(raw_class_name)
            confidence = float(predictions[0][i])

            description_info = CLASS_DESCRIPTIONS.get(key_for_description_lookup, {
                "name": f"Unknown Class: {raw_class_name}",
                "description": "No detailed description available for this specific class. Please consult a medical professional for more information.",
                "symptoms": "No specific symptoms information available.",
                "precautions": "No specific precautions information available.",
                "treatment": "No specific treatment information available."
            })

            results.append({
                "original_class_id": int(i),
                "raw_class_name": raw_class_name,
                "predicted_name": description_info["name"],
                "description": description_info["description"],
                "symptoms": description_info["symptoms"],
                "precautions": description_info["precautions"],
                "treatment": description_info["treatment"],
                "confidence": round(confidence, 4)
            })

        return {
            "filename": file.filename,
            "predictions": results
        }

    except Image.UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not process image. The file might be corrupted or in an unsupported format."
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during prediction processing. Please try again or contact support. Error: {e}"
        )


# --- Root Endpoint (Basic health check / Welcome message) ---
@app.get("/", summary="API Root", response_description="Welcome message and API status.")
async def root():
    """
    A simple endpoint to confirm the API is running.
    """
    return {"message": "Medical Image Classification API is running and ready to receive image predictions!"}

# --- Main entry point for running the application ---
if __name__ == "__main__":
    import uvicorn
    # Run the Uvicorn server.
    uvicorn.run(app, host="0.0.0.0", port=8000)