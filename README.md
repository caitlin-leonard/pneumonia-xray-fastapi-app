# Pneumonia X-Ray Classifier (FastAPI)

A simple CNN-based chest X-ray classifier deployed with **FastAPI**.  
Classifies images as **Normal** or **Pneumonia** via a web UI and REST API.

## Tech Stack
- Python
- FastAPI + Uvicorn
- PyTorch / TensorFlow
- HTML, JavaScript

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload
Open:
http://127.0.0.1:8000
http://127.0.0.1:8000/docs
F
