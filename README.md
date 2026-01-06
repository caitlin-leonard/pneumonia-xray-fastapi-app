# Pneumonia X-Ray Classifier (FastAPI)

Small FastAPI project that trains a convolutional neural network on chest X-ray images and exposes a simple web interface + API to classify images as **Normal** or **Pneumonia**.[web:214][web:218]

---

## Project overview

- `dataset.py` loads and prepares the chest X-ray dataset for training.  
- `train_model.py` trains a CNN model on the dataset and saves the weights locally.  
- `main.py` runs a FastAPI server that loads the trained model and exposes a prediction endpoint.  
- `static/index.html` and `static/script.js` provide a minimal frontend to upload an X-ray image and display the prediction in the browser.[web:214][web:216]

---

## Tech stack

- **Python**  
- **FastAPI** for the API backend.[web:195][web:226]  
- **Uvicorn** as the ASGI server.  
- **PyTorch** or TensorFlow/Keras for the CNN.  

---

## Files and folders

- `api/` (if present): extra API-related modules.  
- `main.py`: FastAPI application entry point.  
- `dataset.py`: dataset loading / preprocessing utilities.  
- `train_model.py`: training script for the pneumonia classifier.  
- `static/index.html`: web page UI for uploading an X-ray image.  
- `static/script.js`: JavaScript to call the API and show the prediction.  
- `README.md`: project description and usage.

Large files like the dataset and trained model weights are kept out of the repo to avoid GitHub size limits.[web:186][web:187]

---

## How to run on macOS

# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (if needed)
python train_model.py

# 4. Start the FastAPI server
uvicorn main:app --reload

# 5. Open in your browser:
#    http://127.0.0.1:8000       -> web UI
#    http://127.0.0.1:8000/docs  -> API docs


