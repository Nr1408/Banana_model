from fastapi import FastAPI, UploadFile, File
import uvicorn
import torch
import io
from PIL import Image
from torchvision import transforms

app = FastAPI()

# 1. Load Your Model (CPU mode)
try:
    # UPDATED: Loads 'model.pt' specifically
    model = torch.jit.load("model.pt", map_location=torch.device('cpu')).eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# 2. Class Names (Exact order from training)
class_names = [
    "Anthracnose", 
    "Banana Fruit-Scarring Beetle", 
    "Banana Skipper Damage", 
    "Banana Split Peel", 
    "Black and Yellow Sigatoka", 
    "Chewing insect damage on banana leaf", 
    "Healthy Banana", 
    "Healthy Banana leaf", 
    "Panama Wilt Disease"
]

# 3. Preprocessing (Matches your Marathon training)
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"message": "Leaflens API is Running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Predict
    img_t = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(img_t)
        probs = torch.nn.functional.softmax(logits[0], dim=0)
        
    # Get Result
    top_prob, top_idx = torch.max(probs, 0)
    prediction = class_names[top_idx.item()]
    confidence = float(top_prob.item())
    
    return {
        "prediction": prediction,
        "confidence": confidence
    }

# --- REPLACE THE BOTTOM OF main.py WITH THIS ---
if __name__ == "__main__":
    import os
    # Get the PORT from Railway, or default to 8080 if running locally
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)