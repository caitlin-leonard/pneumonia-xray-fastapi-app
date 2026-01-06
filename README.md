# Pneumonia X-Ray Classifier (FastAPI)

<img width="555" height="635" alt="Screenshot 2026-01-06 at 6 13 37 PM" src="https://github.com/user-attachments/assets/ef20d471-d1f8-4e39-a867-6e496c94c383" />

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

