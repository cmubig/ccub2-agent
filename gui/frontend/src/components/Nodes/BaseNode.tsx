'use client'

import React, { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeStatus } from '@/lib/types'

interface BaseNodeData {
  label: string
  status: NodeStatus
  nodeType: string
  [key: string]: any
}

const statusIcons = {
  pending: '⏸',
  processing: '🔄',
  completed: '✅',
  error: '❌',
}

const statusColors = {
  pending: 'text-status-pending border-border-dark',
  processing: 'text-status-processing border-border-accent animate-pulse',
  completed: 'text-status-completed border-border-light',
  error: 'text-status-error border-status-error',
}

function BaseNode({ data, selected }: NodeProps<BaseNodeData>) {
  const { label, status, nodeType } = data

  // Check if this is an image-generating node
  const isImageNode = nodeType === 't2i_generator' || nodeType === 'i2i_editor'

  return (
    <div
      className={`
        min-w-[160px] px-4 py-3
        bg-node-bg border-2 rounded-md
        transition-all duration-200
        ${statusColors[status]}
        ${selected ? 'shadow-node-active' : 'shadow-node'}
        hover:bg-node-hover
      `}
    >
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 !bg-white border-2 border-border-dark"
      />

      {/* Image Node with Icon Placeholder */}
      {isImageNode ? (
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <div className="flex-1">
              <div className="text-sm font-medium text-text-primary">{label}</div>
              <div className="text-xs text-text-tertiary mt-0.5">{nodeType}</div>
            </div>
            <div className="text-lg">{statusIcons[status]}</div>
          </div>

          {/* Image Placeholder */}
          <div className="w-64 h-64 bg-neutral-800 border border-neutral-700 rounded flex items-center justify-center">
            {data.image_base64 ? (
              <img
                src={`data:image/png;base64,${data.image_base64}`}
                alt="Generated"
                className="w-full h-full object-cover rounded"
              />
            ) : (
              <svg
                className="w-16 h-16 text-neutral-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
            )}
          </div>
        </div>
      ) : (
        <div className="flex items-center justify-between gap-2">
          <div className="flex-1">
            <div className="text-sm font-medium text-text-primary">{label}</div>
            <div className="text-xs text-text-tertiary mt-0.5">{nodeType}</div>
          </div>

          <div className="text-lg">{statusIcons[status]}</div>
        </div>
      )}

      {status === 'processing' && (
        <div className="mt-2">
          <div className="w-full h-1 bg-surface-dark rounded-full overflow-hidden">
            <div className="h-full bg-white animate-pulse" style={{ width: '100%' }} />
          </div>
        </div>
      )}

      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 !bg-white border-2 border-border-dark"
      />
    </div>
  )
}

export default memo(BaseNode)
