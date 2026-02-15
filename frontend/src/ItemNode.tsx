import { memo } from 'react';
import { Handle, Position } from 'reactflow';

interface ItemNodeData {
  label: string;
  cluster: string;
}

function ItemNode({ data }: { data: ItemNodeData }) {
  return (
    <div
      style={{
        padding: '15px',
        borderRadius: '50%',
        background: 'white',
        border: '3px solid #2196F3',
        width: '60px',
        height: '60px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '16px',
        fontWeight: 'bold',
        cursor: 'grab',
        boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
      }}
    >
      {data.label}
    </div>
  );
}

export default memo(ItemNode);
