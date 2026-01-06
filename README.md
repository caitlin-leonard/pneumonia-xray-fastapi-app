Project overview
dataset.py loads and prepares the chest X-ray dataset for training.

train_model.py trains a CNN model on the dataset and saves the weights locally.

main.py runs a FastAPI server that loads the trained model and exposes a prediction endpoint.

static/index.html and static/script.js provide a minimal frontend to upload an X-ray image and display the prediction in the browser.
​

Tech stack
Python

FastAPI for the API backend.
​

Uvicorn as the ASGI server.

PyTorch or TensorFlow/Keras for the CNN.

Files and folders
api/ (if present): extra API-related modules.

main.py: FastAPI application entry point.

dataset.py: dataset loading / preprocessing utilities.

train_model.py: training script for the pneumonia classifier.

static/index.html: web page UI for uploading an X-ray image.

static/script.js: JavaScript to call the API and show the prediction.

README.md: project description and usage.

Large files like the dataset and trained model weights are kept out of the repo to avoid GitHub size limits.
​

How to run on macOS
bash
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
API endpoints
GET / – serves index.html from the static folder.

POST /predict – accepts an image and returns JSON with predicted label and confidence.
​

Example response:

json
{
  "label": "Pneumonia",
  "confidence": 0.93
}
Disclaimer
Educational and portfolio project only; not for real medical diagnosis or treatment decisions.


