import random
import numpy as np
import vowpal_wabbit_next as vw
from typing import List, Tuple, Dict

def sample_pmf(pmf: List[Tuple[int, float]]) -> Tuple[int, float]:
    """Safely samples an action index from Vowpal Wabbit's PMF output."""
    total = sum(prob for _, prob in pmf) or 1.0
    draw = random.random()
    sum_prob = 0.0
    for index, prob in pmf:
        normalized_prob = prob / total
        sum_prob += normalized_prob
        if sum_prob > draw:
            return index, normalized_prob
    return pmf[-1][0], pmf[-1][1] / total

def get_full_pmf(workspace, parser, item_features: np.ndarray, actions: List[dict]) -> List[Tuple[int, float]]:
    """Get complete probability distribution over ALL actions from VW"""
    # Create prediction-only example (no chosen action)
    example_str = tovwitemformat(item_features, actions, None)
    example = parser.parse_line(example_str)
    
    # Get full probability distribution
    full_prediction = workspace.predict_one(example, PredictionType.ACTION_PROBS)
    
    # full_prediction is array of probabilities [p0, p1, p2, ...] for actions[0], actions[1], ...
    pmf = [(i, float(full_prediction[i])) for i in range(len(actions))]
    
    return pmf

def format_vw_adf_string(cm, item_features: np.ndarray, actions: List[dict], cb_label: tuple = None) -> str:
    """
    Builds the multiline string required for VW's cb_explore_adf algorithm.
    cb_label format: (chosen_action_id, cost, probability)
    """
    # 1. Format the shared context (The item we are trying to cluster)
    item_vw_features = cm.np_to_vw(item_features)
    vw_string = f"shared |Item {item_vw_features}\n"
    
    # 2. Format the candidate actions (The available clusters)
    for action in actions:
        action_id = action["id"]
        cluster_vw_features = action["features"]
        
        # 3. Insert the label if we are in the learning phase
        label_str = ""
        if cb_label is not None:
            chosen_id, cost, prob = cb_label
            if action_id == chosen_id:
                label_str = f"0:{cost}:{prob} "
        
        vw_string += f"{label_str}|Cluster {cluster_vw_features}\n"
        
    return vw_string.strip()

def predict_cluster(workspace: vw.Workspace, parser: vw.TextFormatParser, 
                    cm, item_idx: int, item_features: np.ndarray, learn: bool = True) -> Tuple[dict, float, List[dict]]:
    """
    Fetches the available clusters, formats the context, and asks VW to predict the best fit.
    """
    #print("Current cm clusters:", cm.clusters)
    # Note: cm.get_vw_actions automatically seeds a new cluster for this item.
    actions = cm.get_vw_actions(item_idx, item_features)
    #[print("v") for i in range(4)]
    #print("actions:", actions)
    
    # Format and parse
    vw_example_str = format_vw_adf_string(cm, item_features, actions)
    #print("formatted string for pred:", vw_example_str)
    parsed_lines = [parser.parse_line(line) for line in vw_example_str.split("\n")]
    
    # Predict
    pmf = workspace.predict_one(parsed_lines)
    #print("pmf:", pmf)
    
    # Sample the probability mass function to choose an action
    chosen_action_idx, prob = sample_pmf(pmf)
    chosen_action = actions[chosen_action_idx]
    #print("action stuff:", chosen_action, prob)

    is_new = chosen_action.get("is_new", False)
    
    if learn:
        # if is_new:
        #     cluster_id = cm.add_cluster(item_features, chosen_action["id"])
        #     cm.assign_item(item_idx, cluster_id)

        cost = calculate_cost(cm, item_features, chosen_action["id"], is_new)
        
        # Learn with HIGH cost for new clusters to discourage them
        learn_and_update(workspace, parser, cm, item_idx, item_features, 
                        actions, chosen_action["id"], prob, cost, is_new)

    
    return chosen_action, prob, pmf, actions, chosen_action_idx, is_new

def learn_only(workspace: vw.Workspace, parser: vw.TextFormatParser, cm,
               item_features: np.ndarray, actions: List[dict],
               chosen_action_id: str, prob: float, cost: float):
    """
    Teaches VW the cost/reward of an action WITHOUT syncing ClusterManager.
    Used for multi-pass learning where we want to reinforce the signal.
    """
    cb_label = (chosen_action_id, cost, prob)
    vw_example_str = format_vw_adf_string(cm, item_features, actions, cb_label)
    parsed_lines = [parser.parse_line(line) for line in vw_example_str.split("\n")]
    workspace.learn_one(parsed_lines)

def learn_and_update(workspace: vw.Workspace, parser: vw.TextFormatParser, cm,
                     item_idx: int, item_features: np.ndarray, actions: List[dict],
                     chosen_action_id: str, prob: float, cost: float, is_new_cluster: bool):
    """
    Teaches VW the cost/reward of an action, and syncs the ClusterManager state.
    """
    if is_new_cluster:
        _ = cm.add_cluster(item_features, chosen_action_id)

    # 1. Learn in VW
    learn_only(workspace, parser, cm, item_features, actions, chosen_action_id, prob, cost)

    # 2. Sync ClusterManager (Moves item to the chosen cluster)
    cm.assign_item(item_idx, chosen_action_id)

    # # 3. Clean up empty clusters and update medoids based on new assignments
    # cm.balance_clusters()

def apply_human_correction(workspace: vw.Workspace, parser: vw.TextFormatParser, cm, 
                           item_idx: int, correct_cluster_id: str):
    """
    Simulates a human forcing an item into a specific cluster (Mechanism 2/3).
    Applies a maximum negative cost (high reward) to strictly guide the model.
    """
    # Fetch existing features and current state from ClusterManager
    item_features = cm.item_to_embedding[item_idx]
    #actions = cm.get_vw_actions(item_idx)

    # Predict
    chosen_action, _, pmf, actions, _, _ = predict_cluster(workspace, parser, cm, item_idx, item_features, learn=False)

    is_new_correct_cluster = False
    if correct_cluster_id == "new_cluster":
        is_new_correct_cluster = True
        correct_cluster_id = cm.get_new_cluster_id()

    correct_prob = 0.0
    for idx, action in enumerate(actions):
        if action["id"] == correct_cluster_id:
            # pmf is a list of tuples: [(action_idx, probability), ...]
            correct_prob = next(p for i, p in pmf if i == idx)
            break
    
    # VERY HIGH REWARD for user corrections - this is the key to making them stick!
    # Regular predictions use costs in [0, 1.2] range
    # Corrections use -100.0 to make them MUCH more valuable than accumulated evidence
    cost = -5.0
    prob = 1.0  # Force full exploration weight for corrections (not model's prob)

    # # Create new cluster if requested
    # if is_new_correct_cluster:
    #     _ = cm.add_cluster(item_features, correct_cluster_id)

    # Single learning pass with massive -100.0 reward signal
    # This should be strong enough to dominate regular predictions
    learn_and_update(workspace, parser, cm, item_idx, item_features, actions,
                     correct_cluster_id, prob, cost, is_new_correct_cluster)

    return correct_cluster_id

def calculate_cost(cm, item_features, chosen_cluster_id, is_new: bool, new_cluster_penalty=0.3, spatial_scale=0.5, size_scale=0.2):
    """
    Cost function for VW contextual bandit learning.

    **Cost Scale Design:**
    - Regular predictions: [0.0, 1.2] (penalties, model learns to minimize)
      - New cluster: 0.3 penalty
      - Existing cluster: 0.0-1.2 (spatial + size penalties)
    - User corrections: -100.0 (HUGE reward, ~100x more valuable than avoiding penalties)

    This massive gap ensures corrections dominate accumulated evidence from regular predictions.
    If the distance to an existing cluster is too large, the bandit accepts the 0.3 penalty
    and creates a new cluster. Otherwise, it merges into an existing cluster.
    """
    # Handle new cluster case
    if is_new:
        return new_cluster_penalty  # 0.3 penalty for new clusters

    # It joined an existing cluster. Cost is the Euclidean Distance.
    medoid = cm.get_cluster_medoid(chosen_cluster_id)
    spatial_cost = min(np.linalg.norm(item_features - medoid) * spatial_scale, 1.0)  # Cap at 1.0

    cluster_size = cm.clusters[chosen_cluster_id].size
    size_penalty = (cluster_size / len(cm.item_to_cluster)) * size_scale  # Max 0.2

    return spatial_cost + size_penalty  # Range: [0.0, 1.2] but typically [0.0, 0.8]

def get_confidence_margin(pmf, chosen_idx):
    """Calculate margin between chosen action and next best."""
    probs = sorted([(i, p) for i, p in pmf], key=lambda x: x[1], reverse=True)
    if probs[0][0] != chosen_idx:  # Was this exploration noise?
        return 0.0  # No confidence in random exploration!

    if len(probs) > 1:
        return probs[0][1] - probs[1][1]  # Top minus 2nd place
    return 1.0  # Sole option = maximum confidence

def full_cluster_propagation_dash(workspace, parser, cm, items):
    """
    FULL PROPAGATION with MARGIN-BASED confidence. Updates only when
    top prediction beats 2nd-place by >10% margin AND cluster changes.
    """
    item_ids = list(items.keys())
    if not item_ids:
        return items

    print(f"🔄 Propagating {len(item_ids)} points globally...")
    updates_made = 0

    for item_id in item_ids:
        features = np.array(items[item_id]['features'])
        old_cluster = items[item_id]['cluster']
        #print("cm clusters right before propogate:", cm.clusters)
        
        # 1. Get prediction WITH full PMF
        chosen_action, prob, pmf, actions, chosen_idx, is_new_cluster = predict_cluster(
            workspace, parser, cm, item_id, features, learn=False
        )
        updated_cluster = chosen_action["id"]
        
        # 2. MARGIN CHECK: Top must beat 2nd by 10%
        margin = get_confidence_margin(pmf, chosen_idx)

        if (old_cluster != updated_cluster and margin > 0.10):  # 10% lead over next-best
            #print("item", item_id, "clusters before:", cm.clusters)
            # if is_new_cluster:
            #     print("first off, should not be getting here")
            #     print(is_new_cluster, chosen_action)
                
            #     cm.assign_item(item_id, cluster_id)
            #print("item", item_id, "clusters mid:", cm.clusters)

            cost = calculate_cost(cm, features, updated_cluster, is_new_cluster)
            learn_and_update(workspace, parser, cm, item_id, features, 
                           actions, updated_cluster, prob, cost, is_new_cluster)
            #print("item", item_id, "clusters after:", cm.clusters)
            items[item_id]['cluster'] = updated_cluster
            #print("update for", item_id, "to", updated_cluster)
            updates_made += 1
    
    cm.balance_clusters()
    print(f"✅ Propagation complete: {updates_made} points reassigned")
    return items

def full_cluster_propagation(workspace, parser, cm, items: Dict[int, Dict]) -> Dict[int, Dict]:
    """
    Streamlit-optimized propagation: Updates ONLY low-confidence items with >10% margin.
    No print spam, returns updated items dict.
    """
    if not items:
        return items
    
    updates_made = 0
    item_ids = list(items.keys())
    
    # Only process items without clusters or low confidence (faster)
    for item_id in item_ids:
        if item_id not in cm.item_to_cluster or items[item_id].get('confidence', 0) < 0.6:
            features = np.array(items[item_id]['features'])
            old_cluster = items[item_id].get('cluster')
            
            # Get prediction WITHOUT learning
            chosen_action, prob, pmf, actions, chosen_idx, is_new = predict_cluster(
                workspace, parser, cm, item_id, features, learn=False
            )

            margin = get_confidence_margin(pmf, chosen_idx)

            # Update only if confident AND cluster changed
            if (old_cluster != chosen_action["id"] and margin > 0.10):
                cost = calculate_cost(cm, features, chosen_action["id"], is_new)
                learn_and_update(workspace, parser, cm, item_id, features, 
                               actions, chosen_action["id"], prob, cost, is_new)
                
                items[item_id]['cluster'] = chosen_action["id"]
                items[item_id]['confidence'] = prob
                updates_made += 1
    
    cm.balance_clusters()  # Optional - disabled for stable colors
    return items
