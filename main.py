import os
import logging
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline  # Import Hugging Face summarization pipeline
import fitz  # PyMuPDF
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model
import traceback

# Set environment variable to suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN operations

# Check if CUDA is available
if not tf.config.list_physical_devices('GPU'):
    logging.warning("No GPU found. Falling back to CPU.")

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Mount the static directory to serve HTML files
app.mount("/static", StaticFiles(directory="static"), name="static")
model_name = "google-t5/t5-small"  # Specify your model
# Load the Hugging Face summarization pipeline
summarizer = pipeline("summarization", model=model_name)

# Enhanced CRNN Model with BatchNorm and Dropout
def build_advanced_crnn_model(input_shape=(32, 128, 1), num_classes=37):
    inputs = layers.Input(shape=input_shape)

    # Deeper Convolutional layers with Batch Normalization
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Reshape for RNN input
    new_shape = (input_shape[1] // 16, (input_shape[0] // 16) * 256)
    x = layers.Reshape(target_shape=new_shape)(x)

    # Recurrent layers (Bidirectional LSTM) with Dropout
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='ctc_loss_function', metrics=['accuracy'])  # Use CTC loss
    return model


# Build the enhanced CRNN model
model = build_advanced_crnn_model()

# Character set and max label length
characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{};:'\"\\|,.<>?/`~ "
max_text_length = 128


# Enhanced Image Preprocessing Function
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Preprocess the image for the OCR model."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Apply Gaussian Blur to remove noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Adaptive thresholding for binarization
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

    # Resize to match the model input size
    img = cv2.resize(img, (128, 32))

    # Normalize pixel values
    img = img / 255.0

    # Add channel and batch dimensions
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    return img


# OCR Function using the enhanced CRNN model
def decode_predictions(predictions) -> str:
    """Decode the predictions into readable text."""
    decoded = ""
    for pred in predictions:
        for p in pred:
            if p < len(characters):  # Ensure index is within the valid character set
                decoded += characters[p]
    return decoded


def run_ocr(image: np.ndarray) -> str:
    """Run OCR on the given image using TensorFlow."""
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    decoded_text = decode_predictions(np.argmax(predictions, axis=-1))
    return decoded_text


# Function to extract text from PDF using OCR and text extraction
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as pdf_document:
        logging.info(f"Number of pages: {pdf_document.page_count}")
        for page in pdf_document:
            page_text = page.get_text("text")
            if page_text:
                logging.info(f"Extracted text from page {page.number}: {page_text[:50]}...")
                text += page_text + "\n"
            else:
                logging.info(f"No text found on page {page.number}. Attempting OCR.")
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_array = np.array(img)
                page_text = run_ocr(img_array)
                text += page_text + "\n"

    return text.strip()


# Summarization function using Hugging Face model
def summarize_text(text: str, max_length: int = 100) -> str:
    """Summarize the extracted text using a pre-trained model."""
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']


# FastAPI routes
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return open("static/index.html").read()


@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        return JSONResponse(content={"error": "File type not supported."}, status_code=400)

    pdf_path = f"temp_{file.filename}"
    try:
        with open(pdf_path, "wb") as pdf_file:
            content = await file.read()
            pdf_file.write(content)

        extracted_text = extract_text_from_pdf(pdf_path)
        os.remove(pdf_path)

        # Summarize the extracted text
        summary = summarize_text(extracted_text)

        if extracted_text:
            return JSONResponse(content={
                "extracted_text": extracted_text,
                "summary": summary
            })
        else:
            return JSONResponse(content={"error": "No text found in the PDF."}, status_code=404)

    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        logging.error(traceback.format_exc())  # Log the full traceback
        return JSONResponse(content={"error": "An error occurred while processing the PDF."}, status_code=500)
