"""
FastAPI Backend for Cluster Visualization
Provides REST API for VW-based clustering with human corrections
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import vowpal_wabbit_next as vw

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


# Pydantic models
class Item(BaseModel):
    id: int
    features: List[float]
    cluster: str
    metadata: str


class CorrectionRequest(BaseModel):
    item_id: int
    target_cluster: str


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


@app.get("/api/items", response_model=ItemsResponse)
async def get_items():
    """Get all items and cluster information"""
    item_list = [
        Item(
            id=item_id,
            features=data['features'],
            cluster=data['cluster'],
            metadata=data['metadata']
        )
        for item_id, data in items.items()
    ]

    cluster_sizes = {}
    for item in item_list:
        cluster_sizes[item.cluster] = cluster_sizes.get(item.cluster, 0) + 1

    return ItemsResponse(items=item_list, clusters=cluster_sizes)


@app.post("/api/items", response_model=Item)
async def add_item():
    """Add a new random item"""
    item_id = len(items)
    item_features = np.random.uniform(-2, 2, 2)
    metadata = f"Item #{item_id}"

    chosen_action, prob, _, actions, _, _ = predict_cluster(
        workspace, parser, cm, item_id, item_features
    )

    items[item_id] = {
        'features': item_features.tolist(),
        'metadata': metadata,
        'cluster': chosen_action["id"]
    }

    return Item(
        id=item_id,
        features=item_features.tolist(),
        cluster=chosen_action["id"],
        metadata=metadata
    )


@app.post("/api/correct", response_model=ItemsResponse)
async def apply_correction(correction: CorrectionRequest):
    """Apply human correction and propagate"""
    global items
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

    # Global propagation - model should naturally honor correction
    items = full_cluster_propagation_dash(workspace, parser, cm, items)
    cm.balance_clusters()

    # Return updated items
    return await get_items()


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
