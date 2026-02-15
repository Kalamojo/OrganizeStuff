import dash
from dash import dcc, html, Input, Output, callback, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from cluster_manager import ClusterManager
from utils import (predict_cluster, apply_human_correction, 
                  calculate_cost, full_cluster_propagation)
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
    
    # Main Plot
    dcc.Graph(id="cluster-plot", style={'height': '70vh'}),
    
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
    [Output("cluster-plot", "figure"),
     Output("stats", "children"),
     Output("items-store", "data"),
     Output("cluster-legend", "children"),
     Output("refresh-trigger", "data")],  # ← ADD THIS
    [Input("add-item", "n_clicks"),
     Input("recluster-all", "n_clicks"),
     #Input("human-correct", "n_clicks"),
     Input("refresh-trigger", "data")],   # ← ADD THIS
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
    
    # 2. RECLUSTER ALL (no remove_item needed!)
    elif triggered_id == "recluster-all":
        item_ids = list(items.keys())
        for item_id in item_ids:
            item_features = np.array(items[item_id]['features'])
            actions = cm.get_vw_actions(item_id)
            chosen_action, prob, _, actions_out, _, is_new_cluster = predict_cluster(workspace, parser, cm, item_id, item_features, learn=False)
            
            items[item_id]['cluster'] = chosen_action["id"]
        
        cm.balance_clusters()  # Your method cleans up empties!
    
    # EMPTY STATE
    if not items:
        fig = go.Figure()
        fig.update_layout(
            title="Organic Clustering - Add your first item! ✨",
            xaxis_title="Feature 1", yaxis_title="Feature 2", dragmode='lasso'
        )
        return fig, html.Div(["📈 Items: 0 | Clusters: 0"]), items, html.Div("No clusters yet..."), dash.no_update
    
    # PLOT DATA
    plot_data = []
    colors = px.colors.qualitative.Set3
    for item_id, data in items.items():
        cluster_id = data['cluster']
        color_idx = sum(ord(c) for c in cluster_id if c.isalpha()) % len(colors)
        plot_data.append({
            'x': data['features'][0], 'y': data['features'][1],
            'cluster': cluster_id, 'item_id': item_id,
            'metadata': data['metadata']
        })
    
    df = pd.DataFrame(plot_data)
    # In update_dashboard, REPLACE the px.scatter section:
    # Replace your entire plot creation block:
    df['item_id_display'] = df['item_id'].astype(str)  # For text labels

    fig = px.scatter(df, x='x', y='y', color='cluster',
                    hover_data=['item_id', 'metadata'],
                    custom_data=['item_id'],  # For clicks
                    text='item_id_display',   # Show ID on hover
                    color_discrete_sequence=colors,
                    title="Organic Clustering - Clusters Emerge Automatically ✨")

    fig.update_traces(
        customdata=df['item_id'].values.reshape(-1, 1),
        textposition="middle center",
        marker_size=15, 
        marker_line_width=2, 
        marker_line_color='white',
        textfont_size=10
    )
    fig.update_layout(dragmode='lasso', showlegend=True, hovermode='closest')

    
    # STATS & LEGEND
    cluster_stats = df.groupby('cluster').size().sort_values(ascending=False)
    stats_text = [
        f"📈 Items: {len(items)} | Clusters: {len(cluster_stats)}",
        f"⚡ Largest: {cluster_stats.iloc[0] if len(cluster_stats) > 0 else 0} items"
    ]
    
    legend_items = []
    for cluster_id, size in cluster_stats.items():
        color_idx = sum(ord(c) for c in str(cluster_id) if c.isalpha()) % len(colors)
        legend_items.append(html.Div([
            html.Span("⬤", style={'color': colors[color_idx], 'fontSize': 20, 'marginRight': 8}),
            html.Span(f"{cluster_id}: {int(size)} items", style={'fontWeight': 'bold'})
        ], style={'margin': '8px 0', 'padding': '8px', 'backgroundColor': '#f8f9fa', 'borderRadius': 5}))
    
    return fig, html.Div(stats_text), items, html.Div(legend_items), dash.no_update

# INTERACTIVE HUMAN CORRECTION CALLBACKS
@callback(
    Output("selected-points", "data"),
    Input("cluster-plot", "clickData"),
    State("selected-points", "data"),
    prevent_initial_call=True
)
def store_clicked_item(click_data, selected):
    print(f"RAW click_data: {click_data}")  # DEBUG
    
    if not click_data or not click_data.get('points'):
        return selected or {}
    
    point = click_data['points'][0]
    print(f"Point keys: {point.keys()}")  # DEBUG
    
    # TRY ALL POSSIBLE LOCATIONS for item_id
    item_id = None
    
    # Method 1: customdata (most reliable)
    if 'customdata' in point:
        if isinstance(point['customdata'], list) and len(point['customdata']) > 0:
            item_id = point['customdata'][0]
        elif isinstance(point['customdata'], dict):
            item_id = point['customdata'].get('item_id')
    
    # Method 2: text label
    if item_id is None and 'text' in point:
        item_id = point['text']
    
    # Method 3: hovertext
    if item_id is None and 'hovertext' in point:
        item_id = point['hovertext']
    
    # Clean up
    if item_id and str(item_id).lstrip('#').replace('.', '').isdigit():
        item_id = int(float(str(item_id).lstrip('#').replace('.', '')))
        print(f"✅ Extracted item_id: {item_id}")
        return {'item_id': item_id}
    
    print(f"❌ No valid item_id found in point: {point}")
    return selected or {}




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
    [Output("items-store", "data", allow_duplicate=True),
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
        return items_store, html.Div("❌ Select item + target first"), dash.no_update
    
    item_id = selected_point['item_id']
    #print("items before sync:", items)
    items.update({int(k): v for k, v in (items_store or {}).items()})
    #print("items after sync:", items)
    
    if item_id in items:
        print(f"🧑‍🏫 CORRECTING Item {item_id} → {target_cluster}")
        
        # 1. Apply direct correction
        cluster_id = apply_human_correction(workspace, parser, cm, item_id, target_cluster)
        items[item_id]['cluster'] = cluster_id
        
        # 2. GLOBAL PROPAGATION - THE MAGIC!
        print("correction applied")
        print(items)
        items = full_cluster_propagation(workspace, parser, cm, items)
        cm.balance_clusters()
        
        return items, html.Div(f"✅ {item_id} → {cluster_id}! ({len(items)} total)", 
                              style={'color': 'green', 'fontWeight': 'bold'}), 1
    
    return items_store, html.Div("❌ Item not found"), dash.no_update



if __name__ == '__main__':
    print("🚀 Organic Clustering Dashboard Ready!")
    print("💡 Usage: Add items → Click dots → Correct → Watch VW learn!")
    app.run(debug=True)
