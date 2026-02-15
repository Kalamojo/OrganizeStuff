"""
FastAPI Backend for Cluster Visualization
Provides REST API for VW-based clustering with human corrections
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import vowpal_wabbit_next as vw
import os
from sklearn.decomposition import PCA
import requests
from io import BytesIO
from PIL import Image
import torch
import open_clip
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Import existing logic
import sys
sys.path.append('..')
from cluster_manager import ClusterManager
from utils import (
    predict_cluster,
    apply_human_correction,
    full_cluster_propagation_dash
)

app = FastAPI(title="Organic Clustering API")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (same as dash_app.py)
workspace = vw.Workspace([
    "--cb_explore_adf",
    "--epsilon", "0.1",
    "--learning_rate", "1.0",
    "--initial_t", "1.0"
])
parser = vw.TextFormatParser(workspace)
cm = ClusterManager()
items: Dict = {}
corrected_items = set()  # Track all items that have been manually corrected

# Load OpenCLIP model
print("🔄 Loading OpenCLIP model...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model.eval()  # Set to evaluation mode
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)
print(f"✅ OpenCLIP loaded on {device}")

# PCA for dimensionality reduction (initialized after we have some data)
pca = None
embedding_dim = 512  # ViT-B-32 CLIP embedding dimension


# Pydantic models
class Item(BaseModel):
    id: int
    features: List[float]  # 2D projection for visualization
    cluster: str
    metadata: str
    image_url: Optional[str] = None
    full_embedding: Optional[List[float]] = None  # High-dimensional embedding
    url: Optional[str] = None  # Original URL for bookmarks


class CorrectionRequest(BaseModel):
    item_id: int
    target_cluster: str


class ImageEmbedRequest(BaseModel):
    image_url: str
    metadata: Optional[str] = None


class URLEmbedRequest(BaseModel):
    url: str
    metadata: Optional[str] = None


class ItemsResponse(BaseModel):
    items: List[Item]
    clusters: Dict[str, int]  # cluster_id -> size


# API Endpoints

@app.get("/")
async def root():
    return {
        "message": "Organic Clustering API",
        "docs": "/docs",
        "items": len(items),
        "clusters": len(cm.clusters)
    }


def fit_pca_if_needed():
    """Fit PCA when we have enough high-dimensional items"""
    global pca
    if pca is not None or len(items) < 3:
        return

    # Check if we have high-dimensional embeddings
    high_dim_items = [item for item in items.values() if 'full_embedding' in item]
    if len(high_dim_items) < 3:
        return

    # Fit PCA on all full embeddings
    embeddings = np.array([item['full_embedding'] for item in high_dim_items])
    pca = PCA(n_components=2)
    pca.fit(embeddings)
    print(f"✅ PCA fitted on {len(embeddings)} items. Explained variance: {pca.explained_variance_ratio_}")

    # Update all items with 2D projections (for visualization only)
    for item_id, item in items.items():
        if 'full_embedding' in item:
            projection = pca.transform([item['full_embedding']])[0]
            item['features'] = projection.tolist()
            # DON'T overwrite cm.item_to_embedding - it should keep 512D embeddings for clustering


@app.get("/api/items", response_model=ItemsResponse)
async def get_items():
    """Get all items and cluster information"""
    item_list = [
        Item(
            id=item_id,
            features=data['features'],
            cluster=data['cluster'],
            metadata=data['metadata'],
            image_url=data.get('image_url'),
            full_embedding=data.get('full_embedding'),
            url=data.get('url')
        )
        for item_id, data in items.items()
    ]

    cluster_sizes = {}
    for item in item_list:
        cluster_sizes[item.cluster] = cluster_sizes.get(item.cluster, 0) + 1

    return ItemsResponse(items=item_list, clusters=cluster_sizes)


@app.post("/api/embed_image", response_model=Item)
async def embed_image(request: ImageEmbedRequest):
    """Embed an image using local OpenCLIP model"""
    try:
        # Download and preprocess image
        response = requests.get(request.image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')

        # Preprocess for CLIP
        image_tensor = clip_preprocess(image).unsqueeze(0).to(device)

        # Get embedding
        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize

        full_embedding = image_features.cpu().numpy().flatten()
        print(f"📸 Embedded image: {request.image_url[:60]}... → {full_embedding.shape}")

        item_id = len(items)
        metadata = request.metadata or f"Image #{item_id}"

        # Project to 2D if PCA is available, otherwise use placeholder
        if pca is not None:
            features_2d = pca.transform([full_embedding])[0]
        else:
            # Placeholder until we have enough data for PCA
            features_2d = np.random.uniform(-2, 2, 2)

        # Predict cluster using full embedding
        # Note: predict_cluster will store embedding in cm.item_to_embedding
        chosen_action, prob, _, actions, _, _ = predict_cluster(
            workspace, parser, cm, item_id, full_embedding
        )

        # Store item with both embeddings
        items[item_id] = {
            'features': features_2d.tolist(),
            'full_embedding': full_embedding.tolist(),
            'metadata': metadata,
            'cluster': chosen_action["id"],
            'image_url': request.image_url
        }

        # Fit/refit PCA if we have enough items
        fit_pca_if_needed()

        return Item(
            id=item_id,
            features=items[item_id]['features'],
            cluster=chosen_action["id"],
            metadata=metadata,
            image_url=request.image_url,
            full_embedding=full_embedding.tolist()
        )

    except Exception as e:
        print(f"❌ Error embedding image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to embed image: {str(e)}")


@app.post("/api/embed_url", response_model=Item)
async def embed_url(request: URLEmbedRequest):
    """Embed a URL using OpenCLIP text encoder"""
    try:
        # Fetch the webpage
        response = requests.get(request.url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()

        # Parse HTML and extract text
        soup = BeautifulSoup(response.content, 'html5lib')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        # Limit text length (CLIP has max context length)
        text = text[:1000] if len(text) > 1000 else text

        # Get favicon URL
        parsed_url = urlparse(request.url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        favicon_url = f"https://www.google.com/s2/favicons?domain={parsed_url.netloc}&sz=128"

        # Get page title
        title = soup.find('title')
        page_title = title.string if title else parsed_url.netloc

        # Embed text using OpenCLIP
        text_tokens = open_clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        full_embedding = text_features.cpu().numpy().flatten()
        print(f"🔗 Embedded URL: {request.url[:60]}... → {full_embedding.shape}")

        item_id = len(items)
        metadata = request.metadata or page_title

        # Project to 2D if PCA is available
        if pca is not None:
            features_2d = pca.transform([full_embedding])[0]
        else:
            features_2d = np.random.uniform(-2, 2, 2)

        # Predict cluster using full embedding
        chosen_action, prob, _, actions, _, _ = predict_cluster(
            workspace, parser, cm, item_id, full_embedding
        )

        # Store item with both embeddings and URL info
        items[item_id] = {
            'features': features_2d.tolist(),
            'full_embedding': full_embedding.tolist(),
            'metadata': metadata,
            'cluster': chosen_action["id"],
            'url': request.url,
            'favicon_url': favicon_url,
            'type': 'url'
        }

        # Fit/refit PCA if we have enough items
        fit_pca_if_needed()

        return Item(
            id=item_id,
            features=items[item_id]['features'],
            cluster=chosen_action["id"],
            metadata=metadata,
            image_url=favicon_url,  # Use favicon as image
            full_embedding=full_embedding.tolist(),
            url=request.url
        )

    except Exception as e:
        print(f"❌ Error embedding URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to embed URL: {str(e)}")


@app.post("/api/items", response_model=Item)
async def add_item():
    """Add a new random item (512D to match image embeddings)"""
    item_id = len(items)
    # Generate random 512D embedding (same dimension as CLIP)
    full_embedding = np.random.uniform(-1, 1, embedding_dim)
    full_embedding = full_embedding / np.linalg.norm(full_embedding)  # Normalize like CLIP does
    metadata = f"Random #{item_id}"

    # Project to 2D if PCA is available
    if pca is not None:
        features_2d = pca.transform([full_embedding])[0]
    else:
        features_2d = full_embedding[:2]  # Use first 2 dims as placeholder

    chosen_action, prob, _, actions, _, _ = predict_cluster(
        workspace, parser, cm, item_id, full_embedding
    )

    items[item_id] = {
        'features': features_2d.tolist(),
        'full_embedding': full_embedding.tolist(),
        'metadata': metadata,
        'cluster': chosen_action["id"]
    }

    # Fit/refit PCA if needed
    fit_pca_if_needed()

    return Item(
        id=item_id,
        features=items[item_id]['features'],
        cluster=chosen_action["id"],
        metadata=metadata,
        full_embedding=full_embedding.tolist()
    )


@app.post("/api/correct", response_model=ItemsResponse)
async def apply_correction(correction: CorrectionRequest):
    """Apply human correction and propagate"""
    global items, corrected_items
    item_id = correction.item_id
    target_cluster = correction.target_cluster

    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")

    print(f"🧑‍🏫 CORRECTING Item {item_id} → {target_cluster}")

    # Apply correction
    cluster_id = apply_human_correction(
        workspace, parser, cm, item_id, target_cluster
    )
    print("updated item", item_id, "to cluster", cluster_id)
    items[item_id]['cluster'] = cluster_id

    # Track this correction to protect it during future propagations
    corrected_items.add(item_id)

    # Global propagation - skip ALL corrected items to preserve user corrections
    items = full_cluster_propagation_dash(workspace, parser, cm, items, skip_item_ids=corrected_items)

    # Return updated items
    return await get_items()


@app.post("/api/reset")
async def reset_all():
    """Clear all items and clusters"""
    global items, pca, cm, corrected_items
    items = {}
    pca = None
    cm = ClusterManager()  # Fresh cluster manager
    corrected_items = set()  # Clear correction history
    return {"message": "Reset complete", "items": 0, "clusters": 0}


@app.post("/api/recluster", response_model=ItemsResponse)
async def recluster_all():
    """Re-run clustering on all items"""
    global items
    item_ids = list(items.keys())
    for item_id in item_ids:
        item_features = np.array(items[item_id]['features'])
        chosen_action, prob, _, actions_out, _, is_new_cluster = predict_cluster(
            workspace, parser, cm, item_id, item_features, learn=False
        )
        items[item_id]['cluster'] = chosen_action["id"]

    cm.balance_clusters()
    return await get_items()


@app.delete("/api/items")
async def clear_items():
    """Clear all items (for testing)"""
    global items
    items = {}
    cm.clusters.clear()
    return {"message": "All items cleared"}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Organic Clustering API...")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔗 CORS enabled for: http://localhost:5173")
    uvicorn.run(app, host="0.0.0.0", port=8000)
