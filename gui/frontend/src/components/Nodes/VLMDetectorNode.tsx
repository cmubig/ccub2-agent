'use client'

import React, { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { NodeStatus } from '@/lib/types'

interface VLMDetectorNodeData {
  label: string
  status: NodeStatus
  culturalScore?: number
  promptScore?: number
  issues: string[]
  iteration?: number
  severe_count?: number
  moderate_count?: number
  minor_count?: number
  fixed_count?: number
  remaining_count?: number
  new_count?: number
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

function VLMDetectorNode({ data, selected }: NodeProps<VLMDetectorNodeData>) {
  const {
    label,
    status,
    culturalScore,
    promptScore,
    issues,
    iteration,
    severe_count = 0,
    moderate_count = 0,
    minor_count = 0,
    fixed_count = 0,
    remaining_count = 0,
    new_count = 0
  } = data

  return (
    <div
      className={`
        min-w-[220px] px-4 py-3
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

      {/* Header */}
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <div className="text-sm font-medium text-text-primary">🔍 {label}</div>
            {iteration !== undefined && iteration !== null && (
              <span className="px-1.5 py-0.5 text-xs font-medium bg-blue-500/20 text-blue-400 rounded">
                Iter {iteration}
              </span>
            )}
          </div>
          <div className="text-xs text-text-tertiary">Qwen3-VL-8B</div>
        </div>
        <div className="text-lg">{statusIcons[status]}</div>
      </div>

      {/* Scores */}
      {(culturalScore !== null && culturalScore !== undefined) && (
        <div className="space-y-1.5 mt-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-text-secondary">Cultural:</span>
            <span
              className={`font-mono font-medium ${
                culturalScore >= 8 ? 'text-white' : culturalScore >= 6 ? 'text-text-secondary' : 'text-status-error'
              }`}
            >
              {culturalScore.toFixed(1)}/10
            </span>
          </div>

          {promptScore !== null && promptScore !== undefined && (
            <div className="flex items-center justify-between text-xs">
              <span className="text-text-secondary">Prompt:</span>
              <span className="font-mono font-medium text-text-secondary">{promptScore.toFixed(1)}/10</span>
            </div>
          )}

          {/* Severity Breakdown */}
          {issues && issues.length > 0 && (
            <div className="mt-2 pt-2 border-t border-border-dark space-y-1">
              <div className="text-xs text-text-tertiary mb-1">Issue Severity:</div>
              <div className="flex items-center gap-2 text-xs">
                {severe_count > 0 && (
                  <span className="flex items-center gap-1 text-red-400">
                    🔴 {severe_count}
                  </span>
                )}
                {moderate_count > 0 && (
                  <span className="flex items-center gap-1 text-yellow-400">
                    🟠 {moderate_count}
                  </span>
                )}
                {minor_count > 0 && (
                  <span className="flex items-center gap-1 text-blue-400">
                    🟡 {minor_count}
                  </span>
                )}
                {severe_count === 0 && moderate_count === 0 && minor_count === 0 && (
                  <span className="text-text-tertiary">None detected</span>
                )}
              </div>

              {/* Issue Progression */}
              {iteration !== undefined && iteration > 0 && (fixed_count > 0 || remaining_count > 0 || new_count > 0) && (
                <div className="mt-2 pt-2 border-t border-border-dark/50">
                  <div className="text-xs text-text-tertiary mb-1">Issue Progress:</div>
                  <div className="flex items-center gap-2 text-xs flex-wrap">
                    {fixed_count > 0 && (
                      <span className="flex items-center gap-1 text-green-400">
                        ✓ {fixed_count}
                      </span>
                    )}
                    {remaining_count > 0 && (
                      <span className="flex items-center gap-1 text-yellow-400">
                        ⚠ {remaining_count}
                      </span>
                    )}
                    {new_count > 0 && (
                      <span className="flex items-center gap-1 text-blue-400">
                        🔍 {new_count}
                      </span>
                    )}
                  </div>
                </div>
              )}

              <div className="text-xs text-text-secondary mt-2">Click for details</div>
            </div>
          )}
        </div>
      )}

      {status === 'processing' && (
        <div className="mt-2">
          <div className="w-full h-1 bg-surface-dark rounded-full overflow-hidden">
            <div className="h-full bg-white animate-pulse" style={{ width: '100%' }} />
          </div>
          <div className="text-xs text-text-tertiary mt-1 text-center">Analyzing...</div>
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

export default memo(VLMDetectorNode)
