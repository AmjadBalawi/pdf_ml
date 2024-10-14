import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import tensorflow as tf
from tensorflow.keras import layers, Model
import cv2
import numpy as np
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import fitz  # PyMuPDF
from PIL import Image
from transformers import pipeline  # Hugging Face summarization pipeline

# Suppress TensorFlow and Hugging Face warnings
tf.get_logger().setLevel('ERROR')
hf_logging.set_verbosity_error()

app = FastAPI()

# Mount the static directory to serve HTML files
app.mount("/static", StaticFiles(directory="static"), name="static")

model_name = "google-t5/t5-small"  # Specify the summarization model
summarizer = pipeline("summarization", model=model_name)

# Optimized Lightweight CRNN Model
def build_light_crnn_model(input_shape=(32, 128, 1), num_classes=37):
    inputs = layers.Input(shape=input_shape)

    # Lightweight Convolutional layers
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Reshape for RNN input
    new_shape = (input_shape[1] // 8, (input_shape[0] // 8) * 64)
    x = layers.Reshape(target_shape=new_shape)(x)

    # Smaller Recurrent layers (Bidirectional LSTM)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Build the lightweight CRNN model
model = build_light_crnn_model()

# Preprocess the image for OCR
def preprocess_image(img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Apply Gaussian Blur to remove noise
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = cv2.resize(img, (128, 32))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Decode predictions from the model output
def decode_predictions(predictions) -> str:
    decoded = ""
    for pred in predictions:
        for p in pred:
            if p < len(characters):
                decoded += characters[p]
    return decoded

# Run OCR on an image
def run_ocr(image: np.ndarray) -> str:
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    decoded_text = decode_predictions(np.argmax(predictions, axis=-1))
    return decoded_text

# Extract text from PDF using OCR and text extraction
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

# Summarize extracted text using Hugging Face model
def summarize_text(text: str, max_length: int = 100) -> str:
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

        summary = summarize_text(extracted_text)  # Summarize the extracted text

        if extracted_text:
            return JSONResponse(content={
                "extracted_text": extracted_text,
                "summary": summary
            })
        else:
            return JSONResponse(content={"error": "No text found in the PDF."}, status_code=404)

    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return JSONResponse(content={"error": "An error occurred while processing the PDF."}, status_code=500)
