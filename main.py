from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import torch
import io
import os
from PIL import Image
from torchvision import transforms

app = FastAPI()

# --- NEW: Serve the static folder ---
# This allows the HTML file to be accessed
app.mount("/static", StaticFiles(directory="static"), name="static")

# 1. Load Model
try:
    model = torch.jit.load("model.pt", map_location=torch.device('cpu')).eval()
    print("Model loaded!")
except Exception as e:
    print(f"Error loading model: {e}")

class_names = [
    "Anthracnose", "Banana Fruit-Scarring Beetle", "Banana Skipper Damage",
    "Banana Split Peel", "Black and Yellow Sigatoka", 
    "Chewing insect damage on banana leaf", "Healthy Banana", 
    "Healthy Banana leaf", "Panama Wilt Disease"
]

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- UPDATED: Serve the HTML file at the home page ---
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    img_t = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(img_t)
        probs = torch.nn.functional.softmax(logits[0], dim=0)
        
    top_prob, top_idx = torch.max(probs, 0)
    prediction = class_names[top_idx.item()]
    confidence = float(top_prob.item())
    
    return {"prediction": prediction, "confidence": confidence}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)