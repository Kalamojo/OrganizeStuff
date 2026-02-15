from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import os
import requests
from io import BytesIO
from PIL import Image
import onnxruntime as ort

app = FastAPI(title="Embedding API")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the quantized ONNX model
print("🔄 Loading Quantized ONNX CLIP model...")
model_path = os.path.join(os.path.dirname(__file__), 'clip_model', 'clip_quantized.onnx')
ort_session = ort.InferenceSession(model_path)
print(f"✅ ONNX model loaded from {model_path}")



# Pydantic models
class Item(BaseModel):
    id: int
    features: List[float]  # 2D projection for visualization
    metadata: str
    image_url: Optional[str] = None
    full_embedding: Optional[List[float]] = None  # High-dimensional embedding
    url: Optional[str] = None  # Original URL for bookmarks

class ImageEmbedRequest(BaseModel):
    image_url: str
    metadata: Optional[str] = None

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Embedding API",
        "docs": "/docs",
    }

@app.post("/api/embed_image", response_model=Item)
async def embed_image(request: ImageEmbedRequest):
    """Embed an image using the quantized ONNX CLIP model"""
    try:
        # Download and preprocess image
        response = requests.get(request.image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Preprocess for CLIP (from open_clip implementation)
        image = image.resize((224, 224), Image.BICUBIC)
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array([0.26862954, 0.26130258, 0.27577711])
        image = image.transpose(2, 0, 1)
        image_tensor = np.expand_dims(image, axis=0)

        # Get embedding from ONNX model
        ort_inputs = {ort_session.get_inputs()[0].name: image_tensor}
        ort_outs = ort_session.run(None, ort_inputs)
        image_features = ort_outs[0]
        
        # Normalize
        norm = np.linalg.norm(image_features, axis=1, keepdims=True)
        full_embedding = (image_features / norm).flatten()

        print(f"📸 Embedded image: {request.image_url[:60]}... → {full_embedding.shape}")

        # The API is now stateless. It does not store the item.
        # It returns an Item object with a temporary ID.
        # The frontend is responsible for managing the items and their state.
        return Item(
            id=-1, # Temporary ID
            features=[0.0, 0.0],
            metadata=request.metadata or f"Image: {request.image_url[:30]}...",
            image_url=request.image_url,
            full_embedding=full_embedding.tolist()
        )

    except Exception as e:
        print(f"❌ Error embedding image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to embed image: {str(e)}")



