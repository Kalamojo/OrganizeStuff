from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import os
import requests
from io import BytesIO
from PIL import Image
import onnxruntime as ort
from . import prepare_model

# --- Model Preparation ---
# The model is now prepared during the build step (see build.sh)
MODEL_DIR_NAME = "clip_model"
QUANTIZED_MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_DIR_NAME, "clip_quantized.onnx")

# --- FastAPI App ---
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
ort_session = ort.InferenceSession(QUANTIZED_MODEL_PATH)
print(f"✅ ONNX model loaded from {QUANTIZED_MODEL_PATH}")

# Pydantic models
class Item(BaseModel):
    id: int
    features: List[float]
    metadata: str
    image_url: Optional[str] = None
    full_embedding: Optional[List[float]] = None
    url: Optional[str] = None

class ImageEmbedRequest(BaseModel):
    image_url: str
    metadata: Optional[str] = None

class UrlEmbedRequest(BaseModel):
    url: str
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
        response = requests.get(request.image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        image = image.resize((224, 224), Image.BICUBIC)
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array([0.26862954, 0.26130258, 0.27577711])
        image = image.transpose(2, 0, 1)
        image_tensor = np.expand_dims(image, axis=0)

        ort_inputs = {ort_session.get_inputs()[0].name: image_tensor}
        ort_outs = ort_session.run(None, ort_inputs)
        image_features = ort_outs[0]
        
        norm = np.linalg.norm(image_features, axis=1, keepdims=True)
        full_embedding = (image_features / norm).flatten()

        print(f"📸 Embedded image: {request.image_url[:60]}... → {full_embedding.shape}")

        return Item(
            id=-1,
            features=[0.0, 0.0],
            metadata=request.metadata or f"Image: {request.image_url[:30]}...",
            image_url=request.image_url,
            full_embedding=full_embedding.tolist()
        )

    except Exception as e:
        print(f"❌ Error embedding image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to embed image: {str(e)}")

@app.post("/api/embed_url", response_model=Item)
async def embed_url(request: UrlEmbedRequest):
    """(Placeholder) Embed a URL"""
    try:
        # In a real implementation, you would fetch the URL content,
        # extract text, and use a text embedding model.
        # For now, we generate a random vector to make the workflow functional.
        embedding_size = 512 # Matching CLIP ViT-B/32 embedding size
        full_embedding = np.random.rand(embedding_size).astype(np.float32)
        norm = np.linalg.norm(full_embedding)
        full_embedding = (full_embedding / norm).flatten()

        print(f"🔗 (Placeholder) Embedded URL: {request.url[:60]}...")

        return Item(
            id=-1,
            features=[0.0, 0.0],
            metadata=request.metadata or f"URL: {request.url[:30]}...",
            url=request.url,
            full_embedding=full_embedding.tolist()
        )

    except Exception as e:
        print(f"❌ Error embedding url: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to embed url: {str(e)}")
