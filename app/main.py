from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import models, transforms
import torch
import torch.nn as nn
from PIL import Image
import os
import io
import subprocess

# Model config
MODEL_PATH = "model/resnet50_best.pt"
FILE_ID = "1Hsi-5GEzVnx0A68xz4RH7DY1irxR1lhK"
NUM_CLASSES = 7
LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model folder if missing
os.makedirs("model", exist_ok=True)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        subprocess.run([
            "gdown",
            f"https://drive.google.com/uc?id={FILE_ID}",
            "--output", MODEL_PATH
        ])
        print("Model downloaded.")

# Load model
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

download_model()
model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)
        pred_index = torch.argmax(probs, dim=1).item()
        predicted_label = LABELS[pred_index]

    return JSONResponse({"prediction": predicted_label})
