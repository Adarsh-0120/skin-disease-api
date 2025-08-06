# Skin Disease Detection API (FastAPI + PyTorch)

A FastAPI backend that predicts skin disease labels from images using a ResNet-50 model trained on the HAM10000 dataset.

## Features

- Accepts image uploads
- Loads model automatically from Google Drive
- Returns predicted skin lesion label (e.g., melanoma, nv, akiec)

## API Endpoint

### POST /predict

- **Body**: multipart/form-data with key `file`
- **Response**: JSON with predicted class

## Setup

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
