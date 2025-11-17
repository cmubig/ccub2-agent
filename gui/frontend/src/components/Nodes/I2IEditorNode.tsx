'use client'

import React, { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeStatus } from '@/lib/types'

interface I2IEditorNodeData {
  label: string
  status: NodeStatus
  progress?: number
  currentStep?: number
  totalSteps?: number
  iteration?: number
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

function I2IEditorNode({ data, selected }: NodeProps<I2IEditorNodeData>) {
  const { label, status, progress, currentStep, totalSteps, iteration } = data

  const progressPercent = progress ? progress * 100 : 0

  return (
    <div
      className={`
        min-w-[200px] px-4 py-3
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

      <div className="space-y-2">
        {/* Header */}
        <div className="flex items-center justify-between gap-2">
          <div className="flex-1">
            <div className="text-sm font-medium text-text-primary">✏️ {label}</div>
            <div className="text-xs text-text-tertiary">Qwen-Image-Edit</div>
          </div>
          <div className="text-lg">{statusIcons[status]}</div>
        </div>

        {/* Iteration Badge */}
        {iteration !== undefined && iteration > 0 && (
          <div className="flex items-center gap-1.5 px-2 py-1 bg-purple-900/30 border border-purple-500/50 rounded">
            <svg
              className="w-3.5 h-3.5 text-purple-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
            <span className="text-xs font-medium text-purple-200">
              Iteration {iteration}
            </span>
          </div>
        )}

        {/* Image Placeholder */}
        <div className="w-64 h-64 bg-neutral-800 border border-purple-500/30 rounded flex items-center justify-center relative overflow-hidden">
          {data.image_base64 ? (
            <img
              src={`data:image/png;base64,${data.image_base64}`}
              alt="Edited"
              className="w-full h-full object-cover rounded"
            />
          ) : (
            <div className="flex flex-col items-center gap-2">
              <svg
                className="w-16 h-16 text-purple-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                />
              </svg>
              {iteration !== undefined && iteration > 0 && (
                <span className="text-xs text-purple-400 font-medium">Editing...</span>
              )}
            </div>
          )}

          {/* Progress overlay when processing */}
          {status === 'processing' && currentStep && totalSteps && (
            <div className="absolute bottom-0 left-0 right-0 bg-black/80 backdrop-blur-sm p-2">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-purple-300">Step {currentStep}/{totalSteps}</span>
                <span className="font-mono font-medium text-white">{progressPercent.toFixed(0)}%</span>
              </div>
              <div className="w-full h-1.5 bg-surface-dark rounded-full overflow-hidden">
                <div
                  className="h-full bg-purple-400 transition-all duration-300"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 !bg-white border-2 border-border-dark"
      />
    </div>
  )
}

export default memo(I2IEditorNode)
