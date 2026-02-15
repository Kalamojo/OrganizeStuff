import { useCallback, useEffect, useState } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  NodeTypes,
  OnNodeDrag,
  Panel,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { api } from './api';
import type { Item } from './types';
import ItemNode from './ItemNode';
import ClusterGroup from './ClusterGroup';

const nodeTypes: NodeTypes = {
  item: ItemNode,
  cluster: ClusterGroup,
};

// Colors for clusters
const CLUSTER_COLORS = [
  '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
  '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd'
];

function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [items, setItems] = useState<Item[]>([]);
  const [clusters, setClusters] = useState<Record<string, number>>({});
  const [draggedNode, setDraggedNode] = useState<Node | null>(null);

  // Load items from API
  const loadItems = useCallback(async () => {
    try {
      const data = await api.getItems();
      setItems(data.items);
      setClusters(data.clusters);
      updateNodesFromItems(data.items, data.clusters);
    } catch (error) {
      console.error('Failed to load items:', error);
    }
  }, []);

  useEffect(() => {
    loadItems();
  }, [loadItems]);

  // Convert items to React Flow nodes
  const updateNodesFromItems = (items: Item[], clusterSizes: Record<string, number>) => {
    const clusterNodes: Node[] = [];
    const itemNodes: Node[] = [];
    const clusterPositions: Record<string, { x: number; y: number; count: number }> = {};

    // Create cluster group nodes
    Object.keys(clusterSizes).forEach((clusterId, index) => {
      const x = (index % 3) * 400;
      const y = Math.floor(index / 3) * 400;
      const color = CLUSTER_COLORS[index % CLUSTER_COLORS.length];

      clusterPositions[clusterId] = { x, y, count: 0 };

      clusterNodes.push({
        id: clusterId,
        type: 'cluster',
        position: { x, y },
        data: {
          label: clusterId.replace('_', ' ').toUpperCase(),
          size: clusterSizes[clusterId],
          color,
        },
        style: {
          width: 350,
          height: 350,
          backgroundColor: color + '20',
          border: `3px solid ${color}`,
          borderRadius: '15px',
          padding: '20px',
        },
        draggable: false,
      });
    });

    // Create item nodes inside clusters
    items.forEach((item) => {
      const clusterPos = clusterPositions[item.cluster];
      if (!clusterPos) return;

      // Position items in a grid within cluster
      const itemsPerRow = 4;
      const row = Math.floor(clusterPos.count / itemsPerRow);
      const col = clusterPos.count % itemsPerRow;

      itemNodes.push({
        id: String(item.id),
        type: 'item',
        position: {
          x: clusterPos.x + 50 + col * 70,
          y: clusterPos.y + 80 + row * 70,
        },
        data: {
          label: `#${item.id}`,
          cluster: item.cluster,
        },
        draggable: true,
        // Don't set parentNode or extent to allow free dragging between clusters
      });

      clusterPos.count++;
    });

    setNodes([...clusterNodes, ...itemNodes]);
  };

  // Handle adding new item
  const handleAddItem = async () => {
    try {
      await api.addItem();
      await loadItems();
    } catch (error) {
      console.error('Failed to add item:', error);
    }
  };

  // Handle recluster
  const handleRecluster = async () => {
    try {
      await api.reclusterAll();
      await loadItems();
    } catch (error) {
      console.error('Failed to recluster:', error);
    }
  };

  // Handle drag start
  const onNodeDragStart: OnNodeDrag = useCallback((event, node) => {
    if (node.type === 'item') {
      setDraggedNode(node);
    }
  }, []);

  // Handle drag stop - check if dropped on new cluster
  const onNodeDragStop = useCallback(
    async (event, node) => {
      if (node.type !== 'item' || !draggedNode) return;

      // Find which cluster the node is over
      let targetCluster: string | null = null;
      const nodeCenter = {
        x: node.position.x + 30, // half of node width
        y: node.position.y + 30,
      };

      for (const clusterNode of nodes) {
        if (clusterNode.type === 'cluster') {
          const clusterBounds = {
            left: clusterNode.position.x,
            right: clusterNode.position.x + (clusterNode.style?.width as number || 350),
            top: clusterNode.position.y,
            bottom: clusterNode.position.y + (clusterNode.style?.height as number || 350),
          };

          if (
            nodeCenter.x >= clusterBounds.left &&
            nodeCenter.x <= clusterBounds.right &&
            nodeCenter.y >= clusterBounds.top &&
            nodeCenter.y <= clusterBounds.bottom
          ) {
            targetCluster = clusterNode.id;
            break;
          }
        }
      }

      // If dropped on a different cluster, apply correction
      if (targetCluster && targetCluster !== draggedNode.data.cluster) {
        console.log(`Moving item ${node.id} to ${targetCluster}`);
        try {
          await api.applyCorrection(Number(node.id), targetCluster);
          await loadItems(); // Reload to show propagation
        } catch (error) {
          console.error('Failed to apply correction:', error);
        }
      }
      // If dropped outside all clusters, create a new cluster
      else if (!targetCluster) {
        console.log(`Creating new cluster for item ${node.id}`);
        try {
          await api.applyCorrection(Number(node.id), 'new_cluster');
          await loadItems(); // Reload to show new cluster
        } catch (error) {
          console.error('Failed to create new cluster:', error);
        }
      }

      setDraggedNode(null);
    },
    [draggedNode, nodes, loadItems]
  );

  return (
    <div style={{ width: '100vw', height: '100vh', fontFamily: 'system-ui' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeDragStart={onNodeDragStart}
        onNodeDragStop={onNodeDragStop}
        nodeTypes={nodeTypes}
        fitView
        minZoom={0.2}
        maxZoom={1.5}
      >
        <Background />
        <Controls />
        <Panel position="top-left" style={{ margin: 20 }}>
          <div
            style={{
              backgroundColor: 'white',
              padding: '20px',
              borderRadius: '10px',
              boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
            }}
          >
            <h1 style={{ margin: '0 0 20px 0', fontSize: '24px' }}>
              Organic Clustering
            </h1>
            <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
              <button
                onClick={handleAddItem}
                style={{
                  padding: '10px 20px',
                  fontSize: '16px',
                  cursor: 'pointer',
                  backgroundColor: '#4CAF50',
                  color: 'white',
                  border: 'none',
                  borderRadius: '5px',
                }}
              >
                ➕ Add Item
              </button>
              <button
                onClick={handleRecluster}
                style={{
                  padding: '10px 20px',
                  fontSize: '16px',
                  cursor: 'pointer',
                  backgroundColor: '#2196F3',
                  color: 'white',
                  border: 'none',
                  borderRadius: '5px',
                }}
              >
                🔄 Recluster
              </button>
            </div>
            <div style={{ fontSize: '14px', color: '#666' }}>
              <div>📊 Items: {items.length}</div>
              <div>🎯 Clusters: {Object.keys(clusters).length}</div>
              <div style={{ marginTop: '10px', fontStyle: 'italic' }}>
                💡 Drag items between clusters to teach the algorithm!
              </div>
              <div style={{ marginTop: '5px', fontStyle: 'italic' }}>
                ✨ Drag outside clusters to create new ones!
              </div>
            </div>
          </div>
        </Panel>
      </ReactFlow>
    </div>
  );
}

export default App;
