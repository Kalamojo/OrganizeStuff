import { memo } from 'react';

interface ClusterGroupData {
  label: string;
  size: number;
  color: string;
}

function ClusterGroup({ data }: { data: ClusterGroupData }) {
  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        pointerEvents: 'none',
        position: 'relative',
      }}
    >
      {/* Drag handle in corner */}
      <div
        style={{
          position: 'absolute',
          top: '5px',
          right: '5px',
          width: '20px',
          height: '20px',
          cursor: 'move',
          pointerEvents: 'all',
          opacity: 0.3,
          fontSize: '16px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: data.color,
        }}
        title="Drag to move cluster"
      >
        ⠿
      </div>
      <div
        style={{
          fontSize: '20px',
          fontWeight: 'bold',
          color: data.color,
          marginBottom: '10px',
          textShadow: '1px 1px 2px rgba(0,0,0,0.1)',
        }}
      >
        {data.label}
      </div>
      <div
        style={{
          fontSize: '14px',
          color: '#666',
          transition: 'all 0.3s ease-in-out',
        }}
      >
        <span
          style={{
            display: 'inline-block',
            transition: 'transform 0.3s ease-in-out',
          }}
        >
          {data.size} items
        </span>
      </div>
    </div>
  );
}

export default memo(ClusterGroup);
