from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import os
import requests
from io import BytesIO
from PIL import Image
import onnxruntime as ort
from tokenizers import Tokenizer
from bs4 import BeautifulSoup

# --- Model Paths ---
MODEL_DIR = os.path.join(".", "clip_model")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer", "tokenizer.json")
VISION_MODEL_PATH = os.path.join(MODEL_DIR, "clip_vision_quantized.onnx")
TEXT_MODEL_PATH = os.path.join(MODEL_DIR, "clip_text_quantized.onnx")

# --- CLIP Image Normalization Constants ---
CLIP_IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# --- FastAPI App ---
app = FastAPI(title="Embedding API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model and Tokenizer Loading ---
print("🔄 Loading ONNX models and tokenizer...")
vision_session = ort.InferenceSession(VISION_MODEL_PATH)
text_session = ort.InferenceSession(TEXT_MODEL_PATH)
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
# Set the truncation and padding parameters that were in the original tokenizer config
tokenizer.enable_truncation(max_length=77)
tokenizer.enable_padding(pad_id=0, pad_token="<|endoftext|>", length=77)
print("✅ Models and tokenizer loaded.")

# --- Pydantic Models ---
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

# --- Helper Functions ---
def get_text_from_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    # Get text
    text = soup.get_text()
    # Break into lines and remove leading/trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Embedding API", "docs": "/docs"}

@app.post("/api/embed_image", response_model=Item)
async def embed_image(request: ImageEmbedRequest):
    """Embed an image using the quantized ONNX CLIP vision model"""
    try:
        response = requests.get(request.image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        image = image.resize((224, 224), Image.BICUBIC)
        image = np.array(image, dtype=np.float32) / np.float32(255.0)
        image = (image - CLIP_IMAGE_MEAN) / CLIP_IMAGE_STD
        image = image.transpose(2, 0, 1)
        image_tensor = np.expand_dims(image, axis=0).astype(np.float32)

        ort_inputs = {vision_session.get_inputs()[0].name: image_tensor}
        ort_outs = vision_session.run(None, ort_inputs)
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
    """Embed a URL's text content using the quantized ONNX CLIP text model"""
    try:
        # 1. Fetch and parse HTML
        response = requests.get(request.url, timeout=10)
        response.raise_for_status()
        page_text = get_text_from_html(response.text)

        # 2. Tokenize text using the lightweight 'tokenizers' library
        tokens = np.array([tokenizer.encode(page_text).ids])

        # 3. Run inference
        ort_inputs = {text_session.get_inputs()[0].name: tokens}
        ort_outs = text_session.run(None, ort_inputs)
        text_features = ort_outs[0]

        # 4. Normalize
        norm = np.linalg.norm(text_features, axis=1, keepdims=True)
        full_embedding = (text_features / norm).flatten()
        
        print(f"🔗 Embedded URL: {request.url[:60]}... → {full_embedding.shape}")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
