import { useCallback, useState } from 'react';
import ReactFlow, {
  Node,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  NodeTypes,
  NodeDragHandler,
  Panel,
} from 'reactflow';
import 'reactflow/dist/style.css';
import './animations.css';
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
  const [edges, , onEdgesChange] = useEdgesState([]);
  const [items, setItems] = useState<Record<string, Item>>({});
  const [cmState, setCmState] = useState<any>({});
  const [draggedNode, setDraggedNode] = useState<Node | null>(null);
  const [dragStartPos, setDragStartPos] = useState<{ x: number; y: number } | null>(null);
  const [imageUrl, setImageUrl] = useState<string>('');
  const [isAddingImage, setIsAddingImage] = useState(false);
  const [changedItems] = useState<Set<number>>(new Set());

  const updateUIFromState = (newItems: Record<string, Item>, newCmState: any) => {
    const itemArray = Object.values(newItems);
    const clusterSizes = Object.values(newCmState.clusters || {}).reduce((acc: Record<string, number>, c: any) => {
        acc[c.id] = c.size;
        return acc;
    }, {});
    setItems(newItems);
    setCmState(newCmState);
    updateNodesFromItems(itemArray, clusterSizes);
  }

  // Convert items to React Flow nodes
  const updateNodesFromItems = (items: Item[], clusterSizes: Record<string, number>) => {
    const clusterNodes: Node[] = [];
    const itemNodes: Node[] = [];
    const clusterPositions: Record<string, { x: number; y: number; count: number }> = {};

    // Create cluster group nodes with better spacing
    Object.keys(clusterSizes).forEach((clusterId, index) => {
      const x = (index % 3) * 450;
      const y = Math.floor(index / 3) * 450;
      const color = CLUSTER_COLORS[index % CLUSTER_COLORS.length];

      clusterPositions[clusterId] = { x, y, count: 0 };

      // Dynamic cluster size based on number of items
      const itemCount = clusterSizes[clusterId];
      const minSize = 350;
      const sizeIncrement = 40; // Add 40px for every 4 items
      const dynamicSize = Math.max(minSize, minSize + Math.floor(itemCount / 4) * sizeIncrement);

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
          width: dynamicSize,
          height: dynamicSize,
          backgroundColor: color + '20',
          border: `3px solid ${color}`,
          borderRadius: '15px',
          padding: '20px',
          transition: 'all 0.4s ease-in-out',
        },
        draggable: true,
        selectable: true,
      });
    });

    // Create item nodes inside clusters
    items.forEach((item) => {
      const clusterPos = clusterPositions[item.cluster];
      if (!clusterPos) return;

      // Calculate dynamic items per row based on cluster size
      const itemCount = clusterSizes[item.cluster];
      const minSize = 350;
      const sizeIncrement = 40;
      const clusterSize = Math.max(minSize, minSize + Math.floor(itemCount / 4) * sizeIncrement);
      const itemSpacing = 100; // Increased for 90px items
      const clusterPadding = 100; // Left/right padding
      const itemsPerRow = Math.max(3, Math.floor((clusterSize - clusterPadding) / itemSpacing));

      const row = Math.floor(clusterPos.count / itemsPerRow);
      const col = clusterPos.count % itemsPerRow;

      // Add deterministic offset based on item ID to prevent exact stacking
      const offsetX = ((item.id * 7) % 10) - 5;
      const offsetY = ((item.id * 13) % 10) - 5;

      itemNodes.push({
        id: String(item.id),
        type: 'item',
        position: {
          x: 50 + col * 100 + offsetX,
          y: 80 + row * 100 + offsetY,
        },
        data: {
          label: item.metadata || `#${item.id}`,
          cluster: item.cluster,
          image_url: item.image_url,
          url: item.url,
          isChanging: changedItems.has(item.id),
        },
        draggable: true,
        parentNode: item.cluster,
        style: {
          transition: 'all 0.5s ease-in-out',
        },
        // No extent allows dragging between clusters
      });

      clusterPos.count++;
    });

    setNodes([...clusterNodes, ...itemNodes]);
  };

  // Handle adding image item
  const handleAddImage = async () => {
    if (!imageUrl.trim()) {
      alert('Please enter an image URL');
      return;
    }

    setIsAddingImage(true);
    try {
      // 1. Get embedding from Vercel
      const embeddedItem = await api.embedImage(imageUrl);
      embeddedItem.id = Date.now(); // Assign a unique ID on the frontend
      
      // 2. Send to Cloudflare for clustering
      const newState = await api.postToWorker('CLUSTER_ITEM', {
        state: { cm: cmState, items },
        item: embeddedItem,
      });

      updateUIFromState(newState.items, newState.cm);
      setImageUrl('');
    } catch (error) {
      console.error('Failed to add image:', error);
      alert('Failed to add image. Make sure the URL is valid.');
    } finally {
      setIsAddingImage(false);
    }
  };

  // Handle reset
  const handleReset = async () => {
    if (!confirm('Clear all items and start fresh?')) return;
    try {
        const newState = await api.postToWorker('RESET', { state: { cm: cmState, items } });
        updateUIFromState(newState.items, newState.cm);
    } catch (error) {
      console.error('Failed to reset:', error);
    }
  };

  // Handle drag start
  const onNodeDragStart: NodeDragHandler = useCallback((_event, node) => {
    if (node.type === 'item') {
      setDraggedNode(node);
      setDragStartPos({ x: node.position.x, y: node.position.y });
    }
  }, []);

  // Handle drag stop - check if dropped on new cluster
  const onNodeDragStop = useCallback(
    async (_event: React.MouseEvent, node: Node) => {
      if (node.type !== 'item' || !draggedNode) return;

      // Check if node actually moved (not just a click/double-click)
      if (dragStartPos) {
        const distance = Math.sqrt(
          Math.pow(node.position.x - dragStartPos.x, 2) +
          Math.pow(node.position.y - dragStartPos.y, 2)
        );

        // If moved less than 10px, treat as click/double-click, not a drag
        if (distance < 10) {
          setDraggedNode(null);
          setDragStartPos(null);
          return;
        }
      }

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

      const correctionTarget = targetCluster || 'new_cluster';

      try {
        const newState = await api.postToWorker('APPLY_CORRECTION', {
            state: { cm: cmState, items },
            item_id: Number(node.id),
            target_cluster: correctionTarget
        });
        updateUIFromState(newState.items, newState.cm);
      } catch (error) {
        console.error('Failed to apply correction:', error);
      }


      setDraggedNode(null);
      setDragStartPos(null);
    },
    [draggedNode, dragStartPos, nodes, items, cmState]
  );

  return (
    <div style={{ width: '100vw', height: '100vh', fontFamily: 'system-ui', background: '#1a1a1a' }}>
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
        <Background color="#333" gap={16} />
        <Controls />
        <Panel position="top-left" style={{ margin: 20 }}>
          <div
            style={{
              backgroundColor: '#2d2d2d',
              padding: '20px',
              borderRadius: '10px',
              boxShadow: '0 2px 10px rgba(0,0,0,0.5)',
              border: '1px solid #404040',
            }}
          >
            <h1 style={{ margin: '0 0 20px 0', fontSize: '24px', color: '#fff' }}>
              Organic Clustering
            </h1>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginBottom: '15px' }}>
              <div style={{ display: 'flex', gap: '10px' }}>
                <button
                  onClick={handleReset}
                  style={{
                    padding: '10px 20px',
                    fontSize: '14px',
                    cursor: 'pointer',
                    backgroundColor: '#f44336',
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                  }}
                >
                  🗑️ Reset
                </button>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                <label style={{ fontSize: '12px', color: '#b0b0b0' }}>🖼️ Add Image by URL:</label>
                <div style={{ display: 'flex', gap: '5px' }}>
                  <input
                    type="text"
                    value={imageUrl}
                    onChange={(e) => setImageUrl(e.target.value)}
                    placeholder="https://example.com/image.jpg"
                    style={{
                      flex: 1,
                      padding: '8px',
                      fontSize: '14px',
                      border: '1px solid #555',
                      borderRadius: '4px',
                      backgroundColor: '#3d3d3d',
                      color: '#fff',
                    }}
                    onKeyPress={(e) => e.key === 'Enter' && handleAddImage()}
                  />
                  <button
                    onClick={handleAddImage}
                    disabled={isAddingImage}
                    style={{
                      padding: '8px 16px',
                      fontSize: '14px',
                      cursor: isAddingImage ? 'wait' : 'pointer',
                      backgroundColor: isAddingImage ? '#ccc' : '#FF9800',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                    }}
                  >
                    {isAddingImage ? '⏳' : '📸 Add'}
                  </button>
                </div>
              </div>
            </div>
            <div style={{ fontSize: '14px', color: '#b0b0b0' }}>
              <div>📊 Items: {Object.keys(items).length}</div>
              <div>🎯 Clusters: {Object.keys(cmState.clusters || {}).length}</div>
              <div style={{ marginTop: '10px', fontStyle: 'italic', color: '#888' }}>
                💡 Drag items between clusters to teach the algorithm!
              </div>
              <div style={{ marginTop: '5px', fontStyle: 'italic', color: '#888' }}>
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
