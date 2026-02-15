import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

function ItemNode({ data }: NodeProps) {
  const isChanging = data.isChanging || false;

  const handleDoubleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    if (data.url) {
      window.open(data.url, '_blank', 'noopener,noreferrer');
    } else if (data.image_url) {
      window.open(data.image_url, '_blank', 'noopener,noreferrer');
    }
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (data.image_url && !data.url) {
      // Only for actual images, not favicons - open in new tab
      e.preventDefault();
      window.open(data.image_url, '_blank', 'noopener,noreferrer');
    }
  };

  // Truncate long labels
  const truncateLabel = (label: string, maxLength: number = 20) => {
    if (label.length <= maxLength) return label;
    return label.substring(0, maxLength) + '...';
  };

  return (
    <div
      onDoubleClick={handleDoubleClick}
      onContextMenu={handleContextMenu}
      title={data.label} // Show full text on hover
      style={{
        padding: '6px',
        borderRadius: '10px',
        background: isChanging ? '#3d3d00' : '#2d2d2d',
        border: isChanging ? '3px solid #ff9800' : '2px solid #555',
        width: '90px',
        height: '90px',
        maxWidth: '90px',
        maxHeight: '90px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: data.url ? 'pointer' : 'grab',
        boxShadow: isChanging
          ? '0 0 15px rgba(255, 152, 0, 0.6), 0 2px 4px rgba(0,0,0,0.1)'
          : '0 2px 4px rgba(0,0,0,0.1)',
        transition: 'all 0.3s ease-in-out',
        animation: isChanging ? 'pulse 1s ease-in-out infinite' : 'none',
        overflow: 'hidden',
      }}
    >
      {data.image_url ? (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '3px', width: '100%' }}>
          <img
            src={data.image_url}
            alt={data.label}
            style={{
              width: '55px',
              height: '55px',
              objectFit: 'cover',
              borderRadius: '6px',
            }}
          />
          <div
            style={{
              fontSize: '10px',
              color: '#b0b0b0',
              textAlign: 'center',
              width: '100%',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              padding: '0 3px',
            }}
          >
            {truncateLabel(data.label, 15)}
          </div>
        </div>
      ) : (
        <div
          style={{
            fontSize: '12px',
            fontWeight: 'bold',
            textAlign: 'center',
            color: '#e0e0e0',
            width: '100%',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            wordBreak: 'break-word',
            padding: '4px',
          }}
        >
          {truncateLabel(data.label, 18)}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

export default memo(ItemNode);
