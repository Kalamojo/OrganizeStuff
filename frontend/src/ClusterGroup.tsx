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
      }}
    >
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
        }}
      >
        {data.size} items
      </div>
    </div>
  );
}

export default memo(ClusterGroup);
