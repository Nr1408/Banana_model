from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import io
import os
from PIL import Image
from torchvision import transforms

app = FastAPI(title="Banana Disease Detection API")

# --- Serve static files (UI) ---
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Load Model (fail fast if missing) ---
MODEL_PATH = "model.pt"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

model = torch.jit.load(MODEL_PATH, map_location=torch.device("cpu"))
model.eval()
print("✅ Model loaded successfully")

# --- Class labels ---
class_names = [
    "Anthracnose",
    "Banana Fruit-Scarring Beetle",
    "Banana Skipper Damage",
    "Banana Split Peel",
    "Black and Yellow Sigatoka",
    "Chewing insect damage on banana leaf",
    "Healthy Banana",
    "Healthy Banana leaf",
    "Panama Wilt Disease",
]

# --- Image preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# --- Home page (serves UI) ---
@app.get("/")
async def read_index():
    index_path = "static/index.html"
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)

# --- Health check (for cron / keep-alive) ---
@app.get("/health")
def health():
    return {"ok": True}

# --- Prediction endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file")

    image_data = await file.read()

    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_t = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.nn.functional.softmax(logits[0], dim=0)

    top_prob, top_idx = torch.max(probs, 0)

    return {
        "prediction": class_names[top_idx.item()],
        "confidence": round(float(top_prob.item()), 4),
    }
