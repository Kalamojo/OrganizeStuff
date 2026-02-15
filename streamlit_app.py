"""
Interactive Bookmark Clustering UI with Contextual Bandits
Streamlit + Plotly + Vowpal Wabbit
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import random
import vowpal_wabbit_next as vw
from typing import List, Tuple, Dict, Any
from sklearn.decomposition import PCA
import time

from utils import (
        predict_cluster, 
        apply_human_correction, 
        full_cluster_propagation
    )
from cluster_manager import ClusterManager

# ============================================================================
# STREAMLIT UI
# ============================================================================

def generate_random_bookmark(n_dim=100):
    """Generate random bookmark with embedding"""
    return {
        'id': int(time.time() * 1000 + random.randint(0, 999)),
        'title': f"Bookmark {random.randint(1, 1000)}",
        'url': f"https://example.com/{random.randint(1000, 9999)}",
        'features': np.random.rand(n_dim).astype(np.float32),
        'cluster': None
    }

def project_to_2d(embeddings: List[np.ndarray]) -> np.ndarray:
    """PCA projection for visualization"""
    if len(embeddings) < 2:
        return np.random.rand(len(embeddings), 2)
    pca = PCA(n_components=2)
    return pca.fit_transform(embeddings)

def get_cluster_color(cluster_id: str) -> str:
    """Consistent ID-based coloring"""
    hash_val = abs(hash(cluster_id)) % 360
    return f"hsla({hash_val}, 70%, 60%, 0.8)"

@st.cache_resource
def init_vw_workspace():
    """Initialize VW once and reuse across reruns"""
    # Added --epsilon for exploration and -q for feature crossing
    vw_workspace = vw.Workspace([
        "--cb_explore_adf",
        "--epsilon", "0.1",
        "--learning_rate", "1.0",
        "--initial_t", "1.0"
    ])
    parser = vw.TextFormatParser(vw_workspace)
    cm = ClusterManager()
    return vw_workspace, parser, cm

def main():
    st.set_page_config(page_title="Interactive Clustering", layout="wide")
    st.title("🧠 Interactive Bookmark Clustering")
    st.markdown("Drag points to recluster • Watch contextual bandits learn in real-time")

    vw_workspace, parser, cm = init_vw_workspace()
    
    # Initialize session state and VW
    if 'items' not in st.session_state:
        st.session_state['items'] = {}
    
    items = st.session_state['items']
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("➕ Add Random Bookmark"):
            new_item = generate_random_bookmark()
            items[new_item['id']] = new_item
            
            # Auto-cluster new item
            result = predict_cluster(vw_workspace, parser, cm, new_item['id'], 
                                   new_item['features'], learn=True)
            items[new_item['id']]['cluster'] = result[0]['id']
            st.rerun()
    
    with col2:
        if st.button("🔄 Propagate All"):
            with st.spinner(f"Reclustering {len(items)} items..."):
                updated_items = full_cluster_propagation(vw_workspace, parser, cm, items)
            st.session_state['items'] = updated_items
            st.rerun()

    
    with col3:
        if st.button("🗑️ Clear All"):
            st.session_state['items'] = {}
            st.session_state.cm = ClusterManager()
            st.session_state.vw_workspace = vw.Workspace([
                "--cb_explore_adf",
                "--epsilon", "0.1",
                "--learning_rate", "1.0",
                "--initial_t", "1.0"
            ])
            st.session_state.parser = vw.TextFormatParser(st.session_state.vw_workspace)
            st.rerun()
    
    # Main visualization
    if items:
        # Prepare data for plotting
        item_ids = list(items.keys())
        embeddings = [items[iid]['features'] for iid in item_ids]
        proj_2d = project_to_2d(embeddings)
        
        df_plot = pd.DataFrame({
            'x': proj_2d[:, 0],
            'y': proj_2d[:, 1],
            'cluster': [items[iid].get('cluster', 'unclustered') for iid in item_ids],
            'id': item_ids,
            'title': [items[iid].get('title', f'Item {iid}') for iid in item_ids],
            'confidence': [0.5] * len(items)  # Placeholder
        })
        
        # Create interactive scatter plot
        fig = px.scatter(df_plot, x='x', y='y', color='cluster',
                        hover_data=['title'],
                        color_discrete_map={cid: get_cluster_color(cid) for cid in df_plot['cluster'].unique()},
                        title="Drag points to select → Choose new cluster below")
        
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
        fig.update_layout(height=600, showlegend=True)
        
        #selected_points = st.plotly_chart(fig, key="main_plot", return_selection=True)
        event = st.plotly_chart(fig, key="main_plot", on_select="rerun")
        
        # Selection handling
        if event and event.selection.point_indices:
            selected_indices = event.selection.point_indices
            st.subheader(f"📤 Selected {len(selected_indices)} items")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Show selected items
                for point in selected_indices:
                    idx = point
                    item_id = df_plot.iloc[idx]['id']
                    item = items[item_id]
                    st.markdown(f"**{item['title']}**")
                    st.caption(item['url'])
                    st.markdown("---")
            
            with col2:
                available_clusters = list(cm.clusters.keys()) + ["new_cluster"]
                target_cluster = st.selectbox("Move to:", available_clusters, key="target_cluster")
                
                if st.button("✅ Apply Correction", type="primary"):
                    with st.spinner("Applying human corrections..."):
                        for point in selected_indices:
                            idx = point
                            item_id = df_plot.iloc[idx]['id']
                            cluster_id = apply_human_correction(vw_workspace, parser, cm, item_id, target_cluster)
                            items[item_id]['cluster'] = cluster_id
                    
                    st.session_state['items'] = items
                    st.success(f"Moved {len(selected_indices)} items to {target_cluster}")
                    st.rerun()
        
        # Cluster statistics
        st.subheader("📊 Cluster Stats")
        if cm.clusters:
            cluster_stats = []
            total_items = len(items)
            for cid, cluster in cm.clusters.items():
                cluster_stats.append({
                    'Cluster': cid,
                    'Size': cluster.size,
                    '%': f"{cluster.size/total_items*100:.1f}%",
                    'Medoid Distance': f"{np.linalg.norm(cluster.medoid):.3f}"
                })
            
            st.dataframe(pd.DataFrame(cluster_stats), width="stretch")
    
    else:
        st.info("👆 Click **Add Random Bookmark** to get started!")
    
    # Footer
    st.markdown("---")
    st.caption("Powered by Vowpal Wabbit Contextual Bandits + Streamlit")

if __name__ == "__main__":
    main()
