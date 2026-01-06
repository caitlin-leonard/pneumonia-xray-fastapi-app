import io
import os
from typing import List

import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ----- 1. Load model -----
MODEL_PATH = os.path.join("models", "pneumonia_resnet18.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

checkpoint = torch.load(MODEL_PATH, map_location=device)
classes: List[str] = checkpoint["classes"]

model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# ----- 2. FastAPI app -----
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    # serve the main page at /
    return FileResponse("static/index.html")

# ----- 3. Prediction endpoint -----
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    img_t = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)[0]

    conf, idx = torch.max(probs, dim=0)
    predicted_label = classes[idx.item()]
    confidence = float(conf.item())

    prob_dict = {
        label: float(probs[i].item())
        for i, label in enumerate(classes)
    }

    return {
        "prediction": predicted_label,
        "confidence": confidence,
        "probabilities": prob_dict,
    }
