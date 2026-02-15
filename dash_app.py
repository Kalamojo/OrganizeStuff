import dash
from dash import dcc, html, Input, Output, callback, State
import dash_cytoscape as cytoscape
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from cluster_manager import ClusterManager
from utils import (predict_cluster, apply_human_correction,
                  calculate_cost, full_cluster_propagation_dash)
import vowpal_wabbit_next as vw

# Global state - NO initial clusters!
workspace = vw.Workspace([
    "--cb_explore_adf",
    "--epsilon", "0.1",           # Reduced
    "--learning_rate", "1.0",     # Higher = faster adaptation to corrections
    "--initial_t", "1.0"
])
parser = vw.TextFormatParser(workspace)
cm = ClusterManager()
items = {}

# Helper functions for Cytoscape
def items_to_cytoscape_elements(items_dict):
    """Convert items dict to Cytoscape format with edges for visual grouping"""
    nodes = []
    edges = []

    # Group items by cluster
    clusters = {}
    for item_id, data in items_dict.items():
        cluster_id = data['cluster']
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(item_id)

        nodes.append({
            'data': {
                'id': str(item_id),
                'label': f'#{item_id}',
                'cluster': cluster_id
            },
            'classes': cluster_id
        })

    # Add edges between items in same cluster (for visual grouping)
    for cluster_id, members in clusters.items():
        # Connect each item to a few others in the cluster (not fully connected)
        for i, item_id in enumerate(members):
            # Connect to next 2-3 items (circular)
            for j in range(1, min(4, len(members))):
                target_idx = (i + j) % len(members)
                target_id = members[target_idx]
                edges.append({
                    'data': {
                        'source': str(item_id),
                        'target': str(target_id),
                        'cluster': cluster_id
                    },
                    'classes': cluster_id
                })

    return nodes + edges

def generate_cytoscape_stylesheet(clusters):
    """Generate dynamic stylesheet based on active clusters"""
    colors = px.colors.qualitative.Set3

    base_styles = [
        {
            'selector': 'node',
            'style': {
                'content': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'width': '60px',
                'height': '60px',
                'border-width': 2,
                'border-color': '#fff',
                'font-size': '12px',
                'font-weight': 'bold',
                'color': '#fff'
            }
        },
        {
            'selector': ':selected',
            'style': {
                'border-width': 4,
                'border-color': '#FFD700'
            }
        },
        {
            'selector': 'edge',
            'style': {
                'width': 2,
                'opacity': 0.3,
                'curve-style': 'bezier'
            }
        }
    ]

    # Add color for each cluster (both nodes and edges)
    for idx, cluster_id in enumerate(clusters.keys()):
        color = colors[idx % len(colors)]
        base_styles.append({
            'selector': f'.{cluster_id}',
            'style': {
                'background-color': color,
                'line-color': color
            }
        })

    return base_styles

app = dash.Dash(__name__, prevent_initial_callbacks="initial_duplicate")
app.title = "Organic Clustering Dashboard"

app.layout = html.Div([
    html.H1("Organic Clustering Dashboard", 
            style={'textAlign': 'center', 'marginBottom': 20, 'color': '#2c3e50'}),
    
    # Main Controls
    html.Div([
        html.Button("➕ Add Random Item", id="add-item", n_clicks=0, 
                   style={'margin': '5px', 'padding': '12px', 'fontSize': 16}),
        html.Button("🔄 Recluster All", id="recluster-all", n_clicks=0, 
                   style={'margin': '5px', 'padding': '12px', 'fontSize': 16}),
        # html.Button("🧑‍🏫 Human Correction (Demo)", id="human-correct", n_clicks=0,
        #            style={'margin': '5px', 'padding': '12px', 'fontSize': 16}),
        html.Div(id="stats", style={'margin': '10px', 'fontSize': 18, 'fontWeight': 'bold'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': 15}),
    
    # Main Cytoscape Graph
    cytoscape.Cytoscape(
        id='cluster-graph',
        elements=[],
        layout={
            'name': 'cose',  # Force-directed layout (organic bubbles)
            'animate': True,
            'animationDuration': 500,
            'nodeRepulsion': 8000,
            'idealEdgeLength': 100,
            'edgeElasticity': 100,
            'gravity': 1
        },
        style={'width': '100%', 'height': '70vh', 'backgroundColor': '#f8f9fa'},
        stylesheet=[]  # Will be updated dynamically
    ),
    
    # HUMAN CORRECTION PANEL
    html.Div([
        html.H3("🎯 Interactive Correction", style={'color': '#856404'}),
        html.Div(id="click-info", style={'marginBottom': 10}),
        dcc.Dropdown(id="target-cluster-dropdown", placeholder="Click item → Select target...", 
                    style={'width': '300px'}),
        html.Button("✅ Apply Correction", id="apply-correction", n_clicks=0,
                   style={'marginLeft': '10px', 'padding': '8px 16px', 
                         'backgroundColor': '#28a745', 'color': 'white'}),
        html.Div(id="correction-status")
    ], style={'backgroundColor': '#fff3cd', 'padding': '20px', 'borderRadius': 10, 'margin': '20px 0'}),
    
    # Live Cluster Legend
    html.Div([
        html.H3("📊 Live Clusters"),
        html.Div(id="cluster-legend", style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': 10})
    ]),
    
    # Hidden stores
    dcc.Store(id="refresh-trigger", data=0),

    dcc.Store(id="items-store", data=items),
    dcc.Store(id="selected-points", data=[])
], style={'padding': '30px', 'fontFamily': 'system-ui', 'maxWidth': '1400px', 'margin': '0 auto'})

# MAIN DASHBOARD CALLBACK
@callback(
    [Output("cluster-graph", "elements"),
     Output("cluster-graph", "stylesheet"),
     Output("stats", "children"),
     Output("items-store", "data"),
     Output("cluster-legend", "children"),
     Output("refresh-trigger", "data")],
    [Input("add-item", "n_clicks"),
     Input("recluster-all", "n_clicks"),
     Input("refresh-trigger", "data")],
    [State("items-store", "data")]
)
def update_dashboard(add_clicks, recluster_clicks, refresh_trigger, items_store):
    global items
    items.update({int(k): v for k, v in (items_store or {}).items()})

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # 1. ADD NEW ITEM
    if triggered_id == "add-item":
        item_id = len(items)
        item_features = np.random.uniform(-2, 2, 2)
        metadata = f"Item #{item_id}"

        chosen_action, prob, _, actions, _, _ = predict_cluster(workspace, parser, cm, item_id, item_features)

        items[item_id] = {
            'features': item_features.tolist(),
            'metadata': metadata,
            'cluster': chosen_action["id"]
        }

    # 2. RECLUSTER ALL
    elif triggered_id == "recluster-all":
        item_ids = list(items.keys())
        for item_id in item_ids:
            item_features = np.array(items[item_id]['features'])
            actions = cm.get_vw_actions(item_id)
            chosen_action, prob, _, actions_out, _, is_new_cluster = predict_cluster(workspace, parser, cm, item_id, item_features, learn=False)

            items[item_id]['cluster'] = chosen_action["id"]

        cm.balance_clusters()

    # EMPTY STATE
    if not items:
        empty_elements = []
        empty_stylesheet = generate_cytoscape_stylesheet({})
        return empty_elements, empty_stylesheet, html.Div(["📈 Items: 0 | Clusters: 0"]), items, html.Div("No clusters yet..."), dash.no_update
    
    # GENERATE CYTOSCAPE ELEMENTS
    elements = items_to_cytoscape_elements(items)
    stylesheet = generate_cytoscape_stylesheet(cm.clusters)

    # STATS & LEGEND
    cluster_counts = {}
    for item_id, data in items.items():
        cluster_id = data['cluster']
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

    cluster_stats = pd.Series(cluster_counts).sort_values(ascending=False)
    stats_text = [
        f"📈 Items: {len(items)} | Clusters: {len(cluster_stats)}",
        f"⚡ Largest: {cluster_stats.iloc[0] if len(cluster_stats) > 0 else 0} items"
    ]

    colors = px.colors.qualitative.Set3
    legend_items = []
    for cluster_id, size in cluster_stats.items():
        color_idx = sum(ord(c) for c in str(cluster_id) if c.isalpha()) % len(colors)
        legend_items.append(html.Div([
            html.Span("⬤", style={'color': colors[color_idx], 'fontSize': 20, 'marginRight': 8}),
            html.Span(f"{cluster_id}: {int(size)} items", style={'fontWeight': 'bold'})
        ], style={'margin': '8px 0', 'padding': '8px', 'backgroundColor': '#f8f9fa', 'borderRadius': 5}))

    return elements, stylesheet, html.Div(stats_text), items, html.Div(legend_items), dash.no_update

# INTERACTIVE HUMAN CORRECTION CALLBACKS
@callback(
    Output("selected-points", "data"),
    Input("cluster-graph", "tapNodeData"),
    State("selected-points", "data"),
    prevent_initial_call=True
)
def store_clicked_item(tap_node_data, selected):
    print(f"RAW tap_node_data: {tap_node_data}")  # DEBUG

    if not tap_node_data:
        return selected or {}

    # Cytoscape provides clean node data
    item_id = int(tap_node_data.get('id', -1))
    cluster = tap_node_data.get('cluster', 'unknown')

    print(f"✅ Selected node - Item: {item_id}, Cluster: {cluster}")
    return {'item_id': item_id, 'cluster': cluster}




@callback(
    Output("target-cluster-dropdown", "options"),
    Input("items-store", "data")
)
def update_cluster_dropdown(items_store):
    items_dict = items_store or {}
    clusters = list(set(data['cluster'] for data in items_dict.values()))
    return [{'label': c, 'value': c} for c in clusters] + [{'label': '✨ new_cluster', 'value': 'new_cluster'}]

@callback(
    Output("click-info", "children"),
    Input("selected-points", "data")
)
def update_click_info(selected):
    if not selected:
        return html.Div("👆 Click any dot to select an item", style={'color': '#666'})
    return html.Div(f"🎯 Selected: Item {selected.get('item_id', '?')}", 
                   style={'color': 'green', 'fontWeight': 'bold', 'fontSize': 16})

@callback(
    [Output("cluster-graph", "elements", allow_duplicate=True),
     Output("cluster-graph", "stylesheet", allow_duplicate=True),
     Output("items-store", "data", allow_duplicate=True),
     Output("correction-status", "children"),
     Output("refresh-trigger", "data", allow_duplicate=True)],
    Input("apply-correction", "n_clicks"),
    [State("selected-points", "data"), State("target-cluster-dropdown", "value"),
     State("items-store", "data")],
    prevent_initial_call=True
)
def apply_correction(n_clicks, selected_point, target_cluster, items_store):
    global items, workspace, parser, cm

    if not selected_point or not target_cluster:
        return dash.no_update, dash.no_update, items_store, html.Div("❌ Select item + target first"), dash.no_update

    item_id = selected_point['item_id']
    items.update({int(k): v for k, v in (items_store or {}).items()})

    if item_id in items:
        print(f"🧑‍🏫 CORRECTING Item {item_id} → {target_cluster}")

        # 1. Apply direct correction
        cluster_id = apply_human_correction(workspace, parser, cm, item_id, target_cluster)
        items[item_id]['cluster'] = cluster_id

        # 2. GLOBAL PROPAGATION - THE MAGIC!
        print("correction applied")
        print(items)
        items = full_cluster_propagation_dash(workspace, parser, cm, items)
        cm.balance_clusters()

        # 3. Update Cytoscape elements
        elements = items_to_cytoscape_elements(items)
        stylesheet = generate_cytoscape_stylesheet(cm.clusters)

        return elements, stylesheet, items, html.Div(f"✅ {item_id} → {cluster_id}! ({len(items)} total)",
                              style={'color': 'green', 'fontWeight': 'bold'}), 1

    return dash.no_update, dash.no_update, items_store, html.Div("❌ Item not found"), dash.no_update



if __name__ == '__main__':
    print("🚀 Organic Clustering Dashboard Ready!")
    print("💡 Usage: Add items → Click dots → Correct → Watch VW learn!")
    app.run(debug=True)
