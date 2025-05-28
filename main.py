from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List
from PIL import Image
import torch
import timm
from torchvision import transforms as T
import requests
import json
from fastapi.middleware.cors import CORSMiddleware


# Initialize FastAPI app
app = FastAPI(title="Alzheimer MRI Classification API with Gemini AI", version="1.1")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with ["http://localhost:3000"] or your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Config:
    IMAGE_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    CLASS_NAMES = {
        0: "MildDemented",
        1: "ModerateDemented",
        2: "NonDemented",
        3: "VeryMildDemented"
    }
    MODEL_ACCURACY = 0.9961  # Best validation accuracy
    BEST_EPOCH = 39
    GEMINI_API_KEY = "AIzaSyChaQ-R01HT5nYY0VZbruGhhfFLR3qBa00"
    GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("rexnet_150", pretrained=False, num_classes=4)
model.load_state_dict(torch.load("saved_models/rexnet_150_best_alzheimer.pth", map_location=device), strict=True)
model.eval()
model.to(device)

# Prediction function
def predict_with_confidence(image_tensor):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, 1)
        return pred_class.item(), confidence.item()

# Gemini API call
def get_gemini_response(stage: str, bmi: float, symptoms: List[str]) -> str:
    headers = {'Content-Type': 'application/json'}
    prompt = (
        f"You are an expert neurologist specialized in Alzheimer’s."
        f" Based on the Alzheimer stage: {stage}, BMI: {bmi}, and symptoms: {', '.join(symptoms)},"
        f" provide a detailed structured response with: 1. Predicted Symptoms Summary,"
        f" 2. Possible Medication, 3. Treatment Steps, 4. Lifestyle Advice to improve condition."
    )
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    try:
        response = requests.post(Config.GEMINI_URL, headers=headers, json=data)
        if response.status_code == 200:
            res_json = response.json()
            return res_json['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"Gemini API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

# Response Models
class PredictionResponse(BaseModel):
    prediction: str
    class_index: int
    confidence: float
    confidence_percent: str
    ai_advice: str

class ClassInfoResponse(BaseModel):
    classes: Dict[int, str]
    description: str
    num_classes: int
    model_accuracy: float
    accuracy_percent: str

class ErrorResponse(BaseModel):
    error: str
    details: str = None

class ModelMetricsResponse(BaseModel):
    status: str
    best_validation_accuracy: float
    epoch: int

# Root endpoint
@app.get("/", summary="API Health Check")
async def root():
    return {"message": "Alzheimer MRI Classification API", "status": "healthy"}

# Class info endpoint
@app.get("/class_info", response_model=ClassInfoResponse)
async def get_class_info():
    return {
        "classes": Config.CLASS_NAMES,
        "description": "Alzheimer's Disease MRI Classification Classes",
        "num_classes": len(Config.CLASS_NAMES),
        "model_accuracy": round(Config.MODEL_ACCURACY, 4),
        "accuracy_percent": f"{Config.MODEL_ACCURACY * 100:.2f}%"
    }

# Model metrics endpoint
@app.get("/model_metrics", response_model=ModelMetricsResponse)
async def get_model_metrics():
    return {
        "status": "success",
        "best_validation_accuracy": round(Config.MODEL_ACCURACY, 4),
        "epoch": Config.BEST_EPOCH
    }

# Predict endpoint
@app.post("/predict", response_model=PredictionResponse, responses={
    200: {"description": "Successful prediction", "model": PredictionResponse},
    400: {"description": "Invalid input", "model": ErrorResponse},
    500: {"description": "Internal server error", "model": ErrorResponse}
})
async def predict(
    file: UploadFile = File(...),
    bmi: float = Form(...),
    symptoms: List[str] = Form(...)
):
    try:
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(status_code=400, detail="Only JPG/PNG images are supported")

        image = Image.open(file.file).convert("RGB")

        transform = T.Compose([
            T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=Config.MEAN, std=Config.STD)
        ])

        image_tensor = transform(image)
        pred_class, confidence = predict_with_confidence(image_tensor)

        if pred_class not in Config.CLASS_NAMES:
            raise HTTPException(status_code=500, detail="Invalid class predicted")

        stage = Config.CLASS_NAMES[pred_class]
        gemini_output = get_gemini_response(stage, bmi, symptoms)

        return PredictionResponse(
            prediction=stage,
            class_index=pred_class,
            confidence=round(confidence, 4),
            confidence_percent=f"{confidence * 100:.2f}%",
            ai_advice=gemini_output
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
