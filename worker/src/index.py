import json
from typing import Dict, List, Any
import numpy as np
import vowpal_wabbit_next as vw
from cluster_manager import ClusterManager
from utils import predict_cluster, apply_human_correction, full_cluster_propagation_dash

async def handle(request):
    if request.method == 'OPTIONS':
        return response_with_cors(json.dumps({}))

    try:
        body = await request.json()
        action = body.get("action")
        state = body.get("state", {})
        
        # NOTE: The VW model state is NOT persisted between requests.
        # For a real application, you would serialize/deserialize the model 
        # from a KV store here.
        # Example:
        # vw_model_data = await env.KV_STORE.get("vw_model")
        # workspace = vw.Workspace(..., model_data=vw_model_data)
        workspace = vw.Workspace(["--cb_explore_adf", "--epsilon", "0.2", "--learning_rate", "0.5", "--power_t", "0"])
        parser = vw.TextFormatParser(workspace)
        cm = ClusterManager().from_dict(state.get("cm"))
        items = state.get("items", {})

        if action == "GET_ITEMS":
            # In this new setup, the worker is the source of truth for clustered items
            # So we just return the state
            pass

        elif action == "CLUSTER_ITEM":
            item = body["item"]
            item_id = item["id"]
            embedding = np.array(item["full_embedding"])
            
            # This will predict a cluster and update the cluster manager
            chosen_action, _, _, _, _, _ = predict_cluster(workspace, parser, cm, item_id, embedding, learn=True)
            
            items[str(item_id)] = {
                'id': item_id,
                'features': item.get('features', [0.0, 0.0]),
                'metadata': item['metadata'],
                'image_url': item.get('image_url'),
                'full_embedding': embedding.tolist(),
                'cluster': chosen_action['id']
            }


        elif action == "APPLY_CORRECTION":
            item_id = body["item_id"]
            target_cluster = body["target_cluster"]
            
            # This will learn the correction and update the cluster manager
            correct_cluster_id = apply_human_correction(workspace, parser, cm, item_id, target_cluster)
            items[str(item_id)]["cluster"] = correct_cluster_id

        elif action == "RESET":
            cm = ClusterManager()
            items = {}

        else:
            return response_with_cors(json.dumps({"error": "Invalid action"}), status=400)

        # NOTE: Persist the VW model state for true learning.
        # Example:
        # vw_model_data = workspace.get_model()
        # await env.KV_STORE.put("vw_model", vw_model_data)
        
        # The frontend will now be responsible for storing and sending the state
        response_data = {
            "cm": cm.to_dict(),
            "items": items
        }
        
        return response_with_cors(json.dumps(response_data))

    except Exception as e:
        import traceback
        return response_with_cors(json.dumps({"error": str(e), "trace": traceback.format_exc()}), status=500)

def response_with_cors(data, status=200):
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }
    return (data, status, headers)

async def on_fetch(request, env):
    return await handle(request)
