import random
import numpy as np
import vowpal_wabbit_next as vw
from typing import List, Tuple

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
    print("Current cm clusters:", cm.clusters)
    # Note: cm.get_vw_actions automatically seeds a new cluster for this item.
    actions = cm.get_vw_actions(item_idx, item_features)
    [print("v") for i in range(4)]
    print("actions:", actions)
    
    # Format and parse
    vw_example_str = format_vw_adf_string(cm, item_features, actions)
    #print("formatted string for pred:", vw_example_str)
    parsed_lines = [parser.parse_line(line) for line in vw_example_str.split("\n")]
    
    # Predict
    pmf = workspace.predict_one(parsed_lines)
    print("pmf:", pmf)
    
    # Sample the probability mass function to choose an action
    chosen_action_idx, prob = sample_pmf(pmf)
    chosen_action = actions[chosen_action_idx]
    print("action stuff:", chosen_action, prob)

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

def learn_and_update(workspace: vw.Workspace, parser: vw.TextFormatParser, cm, 
                     item_idx: int, item_features: np.ndarray, actions: List[dict], 
                     chosen_action_id: str, prob: float, cost: float, is_new_cluster: bool):
    """
    Teaches VW the cost/reward of an action, and syncs the ClusterManager state.
    """
    if is_new_cluster:
        _ = cm.add_cluster(item_features, chosen_action_id)

    # 1. Learn in VW
    cb_label = (chosen_action_id, cost, prob)
    vw_example_str = format_vw_adf_string(cm, item_features, actions, cb_label)
    #print("formatted string for learn:", vw_example_str)
    parsed_lines = [parser.parse_line(line) for line in vw_example_str.split("\n")]
    workspace.learn_one(parsed_lines)
    
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
    
    # High reward (-1 cost) and 100% deterministic probability for the correction
    cost = -1.0
    prob = correct_prob
    
    print("human doing stuff")

    print("cm clusters in human right before learning:", cm.clusters)
    learn_and_update(workspace, parser, cm, item_idx, item_features, actions, 
                     correct_cluster_id, prob, cost, is_new_correct_cluster)
    print("cm clusters in human right after learning:", cm.clusters)
    return correct_cluster_id

def calculate_cost(cm, item_features, chosen_cluster_id, is_new: bool, new_cluster_penalty=0.2, spatial_scale=0.3, size_scale=0.3):
    """
    The mathematical linchpin. We penalize distance, but we ALSO penalize 
    creating new clusters. If the distance to an existing cluster is > 2.5, 
    the bandit will swallow the penalty and spawn a new cluster. Otherwise, it merges.
    """
    # Handle new cluster case
    if is_new:
        return new_cluster_penalty  # Pure penalty for new cluster

    # It joined an existing cluster. Cost is the Euclidean Distance.
    medoid = cm.get_cluster_medoid(chosen_cluster_id)
    spatial_cost = np.linalg.norm(item_features - medoid) * spatial_scale

    cluster_size = cm.clusters[chosen_cluster_id].size
    size_penalty = (cluster_size / len(cm.item_to_cluster)) * size_scale

    return spatial_cost + size_penalty   # Scale to [-0.1, 0.1] range

def get_confidence_margin(pmf, chosen_idx):
    """Calculate margin between chosen action and next best."""
    probs = sorted([(i, p) for i, p in pmf], key=lambda x: x[1], reverse=True)
    if probs[0][0] != chosen_idx:  # Was this exploration noise?
        return 0.0  # No confidence in random exploration!

    if len(probs) > 1:
        return probs[0][1] - probs[1][1]  # Top minus 2nd place
    return 1.0  # Sole option = maximum confidence

def full_cluster_propagation(workspace, parser, cm, items):
    """
    FULL PROPAGATION with MARGIN-BASED confidence. Updates only when 
    top prediction beats 2nd-place by >15% margin AND cluster changes.
    """
    item_ids = list(items.keys())
    if not item_ids:
        return items
    
    print(f"🔄 Propagating {len(item_ids)} points globally...")
    updates_made = 0
    
    for item_id in item_ids:
        features = np.array(items[item_id]['features'])
        old_cluster = items[item_id]['cluster']
        print("cm clusters right before propogate:", cm.clusters)
        
        # 1. Get prediction WITH full PMF
        chosen_action, prob, pmf, actions, chosen_idx, is_new_cluster = predict_cluster(
            workspace, parser, cm, item_id, features, learn=False
        )
        updated_cluster = chosen_action["id"]
        
        # 2. MARGIN CHECK: Top must beat 2nd by 15%
        margin = get_confidence_margin(pmf, chosen_idx)
        
        if (old_cluster != updated_cluster and margin > 0.15):  # 15% lead over next-best
            print("item", item_id, "clusters before:", cm.clusters)
            # if is_new_cluster:
            #     print("first off, should not be getting here")
            #     print(is_new_cluster, chosen_action)
                
            #     cm.assign_item(item_id, cluster_id)
            print("item", item_id, "clusters mid:", cm.clusters)

            cost = calculate_cost(cm, features, updated_cluster, is_new_cluster)
            learn_and_update(workspace, parser, cm, item_id, features, 
                           actions, updated_cluster, prob, cost, is_new_cluster)
            print("item", item_id, "clusters after:", cm.clusters)
            items[item_id]['cluster'] = updated_cluster
            print("update for", item_id, "to", updated_cluster)
            updates_made += 1
    
    #cm.balance_clusters()
    print(f"✅ Propagation complete: {updates_made} points reassigned (margin > 0.15)")
    return items
