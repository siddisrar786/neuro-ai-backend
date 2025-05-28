# 🧠 Alzheimer MRI Classification API 🔬🩻

> 🎯 A powerful and intelligent FastAPI service for classifying Alzheimer's disease stages using MRI images and providing expert-level advice via Gemini AI.

![Alzheimer MRI](https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/MRI_brain_Alzheimer.jpg/800px-MRI_brain_Alzheimer.jpg)

---

## 🚀 Features

✅ High Accuracy Model (🔥 **99.56%**)  
✅ MRI Image Upload for Alzheimer's Stage Detection  
✅ AI-driven Clinical Advice using **Gemini API**  
✅ BMI & Symptoms-Based Personalized Feedback  
✅ CORS Enabled for Frontend Integration  
✅ Fully documented API endpoints

---

## 🧰 Tech Stack

- **FastAPI** ⚡ (Modern Web Framework)
- **Torch + timm** 🧠 (Deep Learning with ReXNet-150)
- **Gemini API** 🤖 (Google's AI Assistant)
- **PIL / torchvision** 📷 (Image Handling & Transformations)

---

## 📦 Installation

```bash
git clone https://github.com/siddisrar786/neuro-ai-backend
cd neuro-ai-backend
pip install -r requirements.txt
```

---

## ▶️ Running the Server

```bash
uvicorn main:app --reload
```

Then open your browser to [http://localhost:8000/docs](http://localhost:8000/docs) to access the **Swagger UI**! 🎉

---

## 🛠️ API Endpoints

### `/predict` (POST)
Upload an MRI Image (JPG/PNG) and pass BMI and symptoms.

#### 🔽 Request:
- `file`: MRI Image
- `bmi`: Patient's Body Mass Index (float)
- `symptoms`: List of observed symptoms

#### 🔼 Response:
```json
{
  "prediction": "VeryMildDemented",
  "class_index": 3,
  "confidence": 0.991,
  "confidence_percent": "99.10%",
  "ai_advice": "1. Symptoms Summary...\n2. Possible Medications...\n3. Lifestyle Tips..."
}
```

---

### `/class_info` (GET)
Returns details of the 4 Alzheimer stages:
- NonDemented
- VeryMildDemented
- MildDemented
- ModerateDemented

### `/model_metrics` (GET)
Returns model accuracy and the epoch it was trained to achieve that score.

---

## 🧠 Model Info

- **Architecture**: ReXNet-150 (via `timm`)
- **Trained Accuracy**: ✅ **99.56%**
- **Classes**:
  - `0`: MildDemented
  - `1`: ModerateDemented
  - `2`: NonDemented
  - `3`: VeryMildDemented

---

## 🤖 Gemini API Integration

Based on:
- Predicted Stage
- Patient BMI
- Observed Symptoms

You'll receive expert-like structured clinical insights:
- 🧾 Summary of Symptoms  
- 💊 Medications  
- 🩺 Treatment Steps  
- 🏃 Lifestyle Recommendations

---

## 🧪 Example Curl Request

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -F 'file=@brain_mri.jpg' \
  -F 'bmi=25.3' \
  -F 'symptoms=["memory loss", "confusion"]'
```

---

## 🛡️ License

MIT License © 2025 [Isarar Siddique](https://github.com/siddisrar786)

---

## 📬 Contact

For queries or collaboration:
📧 isararsiddique@domain.com  
🌐 [LinkedIn](https://linkedin.com/in/isarar)

---

**Built with ❤️ to support early Alzheimer diagnosis and care.**
