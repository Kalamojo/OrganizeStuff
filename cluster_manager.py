from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

@dataclass
class Cluster:
    id: str
    medoid: np.ndarray  # vector embedding
    members: List[int]  # item indices in this cluster
    size: int = 0

class ClusterManager:
    def __init__(self):
        self.clusters: Dict[str, Cluster] = {}
        self.item_to_cluster: Dict[int, str] = {}
        self.item_to_embedding: Dict[int, np.ndarray] = {}
        self.next_cluster_id = 0
    
    def add_cluster(self, medoid: np.ndarray, cluster_id: Optional[str] = None) -> str:
        """Add new cluster, returns final ID"""
        final_id = cluster_id or self.get_new_cluster_id()
        self.clusters[final_id] = Cluster(final_id, medoid.copy(), [], 0)
        self.next_cluster_id += 1
        return final_id
    
    def assign_item(self, item_idx: int, cluster_id: str):
        """Assign item to cluster, update membership tracking"""
        if item_idx in self.item_to_cluster:
            old_cluster_id = self.item_to_cluster[item_idx].id
            self.clusters[old_cluster_id].members.remove(item_idx)
            self.clusters[old_cluster_id].size -= 1
        
        self.item_to_cluster[item_idx] = self.clusters[cluster_id]
        self.clusters[cluster_id].members.append(item_idx)
        self.clusters[cluster_id].size += 1

    def np_to_vw(self, arr: np.ndarray, name: str = None, precision: int = 6, feature_prefix: str = "f") -> str:
        """Generic NumPy → VW features"""
        arr = arr.flatten()
        
        # Fixed: Use list comprehension directly
        features = ' '.join([f"{feature_prefix}{i}:{val:.{precision}f}" 
                            for i, val in enumerate(arr)])
        
        if name:
            return f"{name}:{features}"
        return features
    
    def get_vw_actions(self, item_idx: int, item_features: Optional[np.ndarray] = None) -> List[dict]:
        actions = []
        
        # Existing clusters (auto-format medoids)
        for cluster_id, cluster in self.clusters.items():
            action_str = self.np_to_vw(cluster.medoid)
            actions.append({'id': cluster_id, 'features': action_str})
        
        if item_idx in self.item_to_cluster and self.item_to_cluster[item_idx].size == 1:
            return actions

        # Adding new item
        if item_idx not in self.item_to_embedding:
            if item_features is None:
                raise Exception("Provide item embeddings for new items")

            self.item_to_embedding[item_idx] = item_features
        elif item_features is not None:
            print("Warning. Item embedding should not be provided if already existing")

        # Adding new cluster
        item_features = self.item_to_embedding[item_idx]
        new_action_str = self.np_to_vw(item_features)
        #cluster_id = self.add_cluster(item_features)
        cluster_id = self.get_new_cluster_id()
        #self.assign_item(item_idx, cluster_id)
        actions.append({'id': cluster_id, 'features': new_action_str, 'is_new': True})
        
        return actions
    
    def update_medoid(self, cluster_id: str, new_medoid: np.ndarray):
        """Update cluster medoid (e.g., recompute average)"""
        self.clusters[cluster_id].medoid = new_medoid.copy()
    
    def get_cluster_medoid(self, cluster_id: str) -> np.ndarray:
        return self.clusters[cluster_id].medoid.copy()

    def balance_clusters(self):
        """Recompute ALL centroids from scratch - simple but O(n)"""
        to_remove = []
        for cluster in self.clusters.values():
            if cluster.members:
                arr_list = []
                for item_idx in cluster.members:
                    arr_list.append(self.item_to_embedding[item_idx])
                member_array = np.stack(arr_list)  # Shape: (n_members, n_dims)
                self.clusters[cluster.id].medoid = np.mean(member_array, axis=0)
            else:
                to_remove.append(cluster.id)

        for cluster_id in to_remove:
            del self.clusters[cluster_id]

    def get_new_cluster_id(self):
        return f"cluster_{self.next_cluster_id}"
