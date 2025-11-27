import os
import shutil
import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
from PIL import Image
import io
import sys

sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.model import PneumoniaClassifier
    from src.feature_extractor import FeatureExtractor
except ImportError:
    from model import PneumoniaClassifier
    from feature_extractor import FeatureExtractor

app = FastAPI()

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_DIR = BASE_DIR.parent / "uploads"
MODEL_DIR = BASE_DIR.parent / "models"

for directory in [STATIC_DIR, TEMPLATES_DIR, UPLOAD_DIR, MODEL_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize model
MODEL_PATH = MODEL_DIR / "pneumonia_rf_model.pkl"
classifier = PneumoniaClassifier(model_path=str(MODEL_PATH))

# Load model if it exists
try:
    if MODEL_PATH.exists():
        classifier.load_model()
        print("Model loaded successfully")
    else:
        print("No pre-trained model found. A new model will be created on first training.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handle image upload and prediction."""
    try:
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Only image files are allowed")
        
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            image = Image.open(file_path).convert('L')
            image = image.resize((224, 224))
            
            features = FeatureExtractor.extract_features(image)
            
            prediction = classifier.predict([features])
            proba = classifier.predict_proba([features])[0]
            
            class_name = classifier.label_encoder.inverse_transform(prediction)[0]
            
            return {
                "filename": file.filename,
                "prediction": class_name,
                "confidence": float(max(proba)),
                "probabilities": {
                    "NORMAL": float(proba[0]),
                    "PNEUMONIA": float(proba[1])
                }
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain_model():
    """Handle model retraining."""
    try:

        if not hasattr(retrain_model, 'X_train') or not hasattr(retrain_model, 'y_train'):
            raise HTTPException(
                status_code=400, 
                detail="Training data not available. Please implement data loading logic."
            )
        
        # Retrain the model
        classifier.train(retrain_model.X_train, retrain_model.y_train)
        
        # Save the retrained model
        classifier.save_model()
        
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "model_path": str(MODEL_PATH)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start():
    """Start the FastAPI server."""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )

if __name__ == "__main__":
    start()