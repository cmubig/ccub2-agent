'use client'

import React from 'react'
import { Node } from 'reactflow'
import { X } from 'lucide-react'

interface NodeDetailPanelProps {
  node: Node
  onClose: () => void
}

export default function NodeDetailPanel({ node, onClose }: NodeDetailPanelProps) {
  const { data } = node

  return (
    <div className="bg-neutral-900 border-2 border-neutral-700 rounded-xl shadow-2xl overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-neutral-800 flex items-center justify-between bg-neutral-950">
        <div>
          <h2 className="text-lg font-medium text-text-primary">{data.label}</h2>
          <p className="text-xs text-text-tertiary mt-0.5">Node ID: {node.id}</p>
        </div>
        <button
          onClick={onClose}
          className="p-1 hover:bg-surface-hover rounded transition-colors"
          aria-label="Close"
        >
          <X className="w-5 h-5 text-text-secondary" />
        </button>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4 max-h-[600px] overflow-y-auto">
        {/* Status */}
        <div>
          <h3 className="text-sm font-medium text-text-secondary mb-2">Status</h3>
          <div className="px-3 py-2 bg-surface-dark border border-border-base rounded">
            <span className={`text-sm font-medium ${getStatusColor(data.status)}`}>
              {data.status.toUpperCase()}
            </span>
          </div>
        </div>

        {/* VLM Detector Specific */}
        {node.id === 'vlm_detector' && data.cultural_score !== undefined && (
          <>
            {/* Iteration Info */}
            {data.iteration !== undefined && (
              <div className="bg-blue-900/20 border border-blue-500/50 rounded-lg p-3">
                <div className="flex items-center gap-2">
                  <svg
                    className="w-5 h-5 text-blue-400"
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
                  <span className="text-sm font-medium text-blue-200">
                    Iteration {data.iteration}
                  </span>
                </div>
              </div>
            )}

            <div>
              <h3 className="text-sm font-medium text-text-secondary mb-2">Quality Scores</h3>
              <div className="space-y-2">
                <div className="px-3 py-2 bg-surface-dark border border-border-base rounded">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-text-secondary">Cultural Accuracy:</span>
                    <span className={`text-sm font-mono font-medium ${
                      (data.cultural_score || 0) >= 8 ? 'text-green-400' :
                      (data.cultural_score || 0) >= 6 ? 'text-yellow-400' : 'text-red-400'
                    }`}>
                      {data.cultural_score?.toFixed(1) || 'N/A'}/10
                    </span>
                  </div>
                  {data.score_adjusted && (
                    <div className="mt-1 text-xs text-orange-400 flex items-center gap-1">
                      <span>⚠</span>
                      <span>Adjusted from {data.original_cultural_score?.toFixed(1)} due to severe issues</span>
                    </div>
                  )}
                </div>
                <div className="px-3 py-2 bg-surface-dark border border-border-base rounded">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-text-secondary">Prompt Adherence:</span>
                    <span className={`text-sm font-mono font-medium ${
                      (data.prompt_score || 0) >= 8 ? 'text-green-400' :
                      (data.prompt_score || 0) >= 6 ? 'text-yellow-400' : 'text-red-400'
                    }`}>
                      {data.prompt_score?.toFixed(1) || 'N/A'}/10
                    </span>
                  </div>
                </div>
              </div>

              {/* Severity Breakdown */}
              {(data.severe_count > 0 || data.moderate_count > 0 || data.minor_count > 0) && (
                <div className="mt-3 px-3 py-2 bg-neutral-800/50 border border-neutral-700/50 rounded">
                  <div className="text-xs text-neutral-400 mb-1.5">Issue Severity Breakdown:</div>
                  <div className="space-y-1">
                    {data.severe_count > 0 && (
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-red-300">🔴 Severe (≥8):</span>
                        <span className="text-red-400 font-medium">{data.severe_count}</span>
                      </div>
                    )}
                    {data.moderate_count > 0 && (
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-orange-300">🟠 Moderate (5-7):</span>
                        <span className="text-orange-400 font-medium">{data.moderate_count}</span>
                      </div>
                    )}
                    {data.minor_count > 0 && (
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-yellow-300">🟡 Minor (&lt;5):</span>
                        <span className="text-yellow-400 font-medium">{data.minor_count}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Score Mismatch Warning */}
              {data.score_mismatch && (
                <div className="mt-3 px-3 py-2 bg-red-900/30 border border-red-500/60 rounded">
                  <div className="flex items-start gap-2">
                    <span className="text-red-400 text-sm">⚠️</span>
                    <div className="flex-1 text-xs text-red-200">
                      <div className="font-semibold mb-1">High Score Inconsistency Detected</div>
                      <div>VLM gave high score ({data.original_cultural_score?.toFixed(1)}) but found {data.severe_count} severe issues. Score adjusted to {data.cultural_score?.toFixed(1)}.</div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Issue Progression */}
            {(data.fixed_count > 0 || data.remaining_count > 0 || data.new_count > 0) && (
              <div>
                <h3 className="text-sm font-medium text-text-secondary mb-2">
                  📊 Issue Progression
                </h3>
                <div className="space-y-2">
                  {data.fixed_count > 0 && (
                    <div className="px-3 py-2 bg-green-900/20 border border-green-500/50 rounded flex items-center justify-between">
                      <span className="text-sm text-green-200">✓ Fixed from previous iteration</span>
                      <span className="text-sm font-mono font-medium text-green-400">{data.fixed_count}</span>
                    </div>
                  )}
                  {data.remaining_count > 0 && (
                    <div className="px-3 py-2 bg-orange-900/20 border border-orange-500/50 rounded flex items-center justify-between">
                      <span className="text-sm text-orange-200">⚠ Still needs fixing</span>
                      <span className="text-sm font-mono font-medium text-orange-400">{data.remaining_count}</span>
                    </div>
                  )}
                  {data.new_count > 0 && (
                    <div className="px-3 py-2 bg-blue-900/20 border border-blue-500/50 rounded flex items-center justify-between">
                      <span className="text-sm text-blue-200">🔍 New issues detected</span>
                      <span className="text-sm font-mono font-medium text-blue-400">{data.new_count}</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {data.issues && data.issues.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-text-secondary mb-2">
                  🔍 Detected Issues ({data.issue_count || data.issues.length})
                </h3>
                <div className="space-y-2 max-h-60 overflow-y-auto">
                  {data.issues.slice(0, 5).map((issue: any, index: number) => {
                    const issueText = typeof issue === 'string' ? issue : JSON.stringify(issue);
                    return (
                      <div key={index} className="px-3 py-2 bg-red-900/20 border border-red-500/50 rounded">
                        <div className="flex gap-2">
                          <span className="text-red-400 text-sm">⚠</span>
                          <span className="text-sm text-red-200">{issueText}</span>
                        </div>
                      </div>
                    );
                  })}
                  {data.issues.length > 5 && (
                    <div className="text-xs text-center text-neutral-500">
                      ... and {data.issues.length - 5} more issues
                    </div>
                  )}
                </div>
                <div className="mt-2 text-xs text-yellow-400 bg-yellow-900/20 border border-yellow-500/50 rounded p-2">
                  💡 These issues will be addressed in the next iteration
                </div>
              </div>
            )}

            {/* Iteration History Timeline */}
            {data.iteration_history && data.iteration_history.length > 1 && (
              <div>
                <h3 className="text-sm font-medium text-text-secondary mb-2">
                  📜 Iteration History ({data.iteration_history.length} iterations)
                </h3>
                <div className="space-y-4 max-h-[600px] overflow-y-auto">
                  {data.iteration_history.map((iter: any, index: number) => (
                    <div
                      key={index}
                      className={`rounded-lg border overflow-hidden ${
                        index === data.iteration_history.length - 1
                          ? 'bg-blue-900/30 border-blue-500/70'
                          : 'bg-surface-dark border-border-base'
                      }`}
                    >
                      {/* Header */}
                      <div className="p-3 border-b border-neutral-700/50">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-semibold text-white">
                            Iteration {iter.iteration}
                            {index === data.iteration_history.length - 1 && (
                              <span className="ml-2 text-xs text-blue-400">(Current)</span>
                            )}
                          </span>
                          <div className="flex gap-3">
                            <span className={`text-xs font-mono ${
                              (iter.vlm_cultural_score || 0) >= 8 ? 'text-green-400' :
                              (iter.vlm_cultural_score || 0) >= 6 ? 'text-yellow-400' : 'text-red-400'
                            }`}>
                              Cultural: {iter.vlm_cultural_score?.toFixed(1) || 'N/A'}/10
                            </span>
                            <span className={`text-xs font-mono ${
                              (iter.vlm_prompt_score || 0) >= 8 ? 'text-green-400' :
                              (iter.vlm_prompt_score || 0) >= 6 ? 'text-yellow-400' : 'text-red-400'
                            }`}>
                              Prompt: {iter.vlm_prompt_score?.toFixed(1) || 'N/A'}/10
                            </span>
                          </div>
                        </div>

                        <div className="grid grid-cols-3 gap-2 text-xs">
                          <div className="flex items-center justify-between">
                            <span className="text-neutral-400">Issues:</span>
                            <span className="text-red-300 font-medium">{iter.vlm_issue_count || 0}</span>
                          </div>
                          {iter.fixed_issues && iter.fixed_issues.length > 0 && (
                            <div className="flex items-center justify-between">
                              <span className="text-neutral-400">Fixed:</span>
                              <span className="text-green-400 font-medium">{iter.fixed_issues.length}</span>
                            </div>
                          )}
                          {iter.remaining_issues && iter.remaining_issues.length > 0 && (
                            <div className="flex items-center justify-between">
                              <span className="text-neutral-400">Remaining:</span>
                              <span className="text-orange-400 font-medium">{iter.remaining_issues.length}</span>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Image Preview */}
                      {iter.image_path && (
                        <div className="p-2 bg-neutral-900/50">
                          <img
                            src={`http://localhost:8000/api/images/serve-by-path?path=${encodeURIComponent(iter.image_path)}`}
                            alt={`Iteration ${iter.iteration} result`}
                            className="w-full h-auto object-contain max-h-48 rounded"
                            onError={(e) => {
                              e.currentTarget.style.display = 'none';
                              const parent = e.currentTarget.parentElement;
                              if (parent) {
                                parent.innerHTML = `<div class="p-2 text-xs text-neutral-500 text-center">Image not available</div>`;
                              }
                            }}
                          />
                        </div>
                      )}

                      {/* Editing Prompt Details */}
                      {iter.editing_prompt && (
                        <details className="border-t border-neutral-700/30">
                          <summary className="px-3 py-2 text-xs text-blue-400 cursor-pointer hover:text-blue-300 hover:bg-neutral-800/30 transition-colors">
                            Show editing prompt used
                          </summary>
                          <div className="px-3 py-2 bg-neutral-800/50 text-xs text-neutral-300 border-t border-neutral-700/30">
                            {iter.editing_prompt}
                          </div>
                        </details>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {/* I2I Editor Specific */}
        {node.id === 'i2i_editor' && (
          <>
            {/* Iteration Info */}
            {data.iteration !== undefined && (
              <div className="bg-purple-900/20 border border-purple-500/50 rounded-lg p-3">
                <div className="flex items-center gap-2">
                  <svg
                    className="w-5 h-5 text-purple-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                  <span className="text-sm font-medium text-purple-200">
                    Editing Iteration {data.iteration}
                  </span>
                </div>
              </div>
            )}

            {/* Applied Prompt */}
            {data.editing_prompt && (
              <div>
                <h3 className="text-sm font-medium text-text-secondary mb-2">
                  ✏️ Applied Edits
                </h3>
                <div className="px-3 py-2 bg-green-900/20 border border-green-500/50 rounded">
                  <p className="text-sm text-green-100">{data.editing_prompt}</p>
                </div>
              </div>
            )}

            {data.currentStep && data.totalSteps && (
              <div>
                <h3 className="text-sm font-medium text-text-secondary mb-2">Progress</h3>
                <div className="px-3 py-2 bg-surface-dark border border-border-base rounded">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-text-secondary">Step:</span>
                    <span className="text-sm font-mono font-medium text-white">
                      {data.currentStep}/{data.totalSteps}
                    </span>
                  </div>
                  <div className="w-full h-2 bg-background-primary rounded-full overflow-hidden">
                    <div
                      className="h-full bg-white transition-all duration-300"
                      style={{ width: `${((data.currentStep / data.totalSteps) * 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Show what was addressed */}
            {data.issues_addressed && data.issues_addressed.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-text-secondary mb-2">
                  ✏️ Issues Addressed
                </h3>
                <div className="space-y-1">
                  {data.issues_addressed.map((issue: string, index: number) => (
                    <div key={index} className="text-xs text-blue-400 flex items-center gap-2">
                      <span>→</span>
                      <span>{issue}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Iteration History for I2I Editor */}
            {data.iteration_history && data.iteration_history.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-text-secondary mb-2">
                  🎨 Editing Progress ({data.iteration_history.length} iterations)
                </h3>
                <div className="space-y-4 max-h-[600px] overflow-y-auto">
                  {data.iteration_history.map((iter: any, index: number) => (
                    <div
                      key={index}
                      className={`rounded-lg border overflow-hidden ${
                        index === data.iteration_history.length - 1
                          ? 'bg-purple-900/30 border-purple-500/70'
                          : 'bg-surface-dark border-border-base'
                      }`}
                    >
                      {/* Header */}
                      <div className="p-3 border-b border-neutral-700/50">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-semibold text-white">
                            Edit {iter.iteration}
                            {index === data.iteration_history.length - 1 && (
                              <span className="ml-2 text-xs text-purple-400">(Latest)</span>
                            )}
                          </span>
                          <div className="flex gap-3">
                            <span className={`text-xs font-mono ${
                              (iter.vlm_cultural_score || 0) >= 8 ? 'text-green-400' :
                              (iter.vlm_cultural_score || 0) >= 6 ? 'text-yellow-400' : 'text-red-400'
                            }`}>
                              Cultural: {iter.vlm_cultural_score?.toFixed(1) || 'N/A'}/10
                            </span>
                            <span className={`text-xs font-mono ${
                              (iter.vlm_prompt_score || 0) >= 8 ? 'text-green-400' :
                              (iter.vlm_prompt_score || 0) >= 6 ? 'text-yellow-400' : 'text-red-400'
                            }`}>
                              Prompt: {iter.vlm_prompt_score?.toFixed(1) || 'N/A'}/10
                            </span>
                          </div>
                        </div>

                        <div className="grid grid-cols-3 gap-2 text-xs">
                          <div className="flex items-center justify-between">
                            <span className="text-neutral-400">Issues:</span>
                            <span className="text-red-300 font-medium">{iter.vlm_issue_count || 0}</span>
                          </div>
                          {iter.fixed_issues && iter.fixed_issues.length > 0 && (
                            <div className="flex items-center justify-between">
                              <span className="text-neutral-400">Fixed:</span>
                              <span className="text-green-400 font-medium">{iter.fixed_issues.length}</span>
                            </div>
                          )}
                          {iter.remaining_issues && iter.remaining_issues.length > 0 && (
                            <div className="flex items-center justify-between">
                              <span className="text-neutral-400">Remaining:</span>
                              <span className="text-orange-400 font-medium">{iter.remaining_issues.length}</span>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Image Preview */}
                      {iter.image_path && (
                        <div className="p-2 bg-neutral-900/50">
                          <img
                            src={`http://localhost:8000/api/images/serve-by-path?path=${encodeURIComponent(iter.image_path)}`}
                            alt={`Edit ${iter.iteration} result`}
                            className="w-full h-auto object-contain max-h-48 rounded"
                            onError={(e) => {
                              e.currentTarget.style.display = 'none';
                              const parent = e.currentTarget.parentElement;
                              if (parent) {
                                parent.innerHTML = `<div class="p-2 text-xs text-neutral-500 text-center">Image not available</div>`;
                              }
                            }}
                          />
                        </div>
                      )}

                      {/* Editing Prompt Details */}
                      {iter.editing_prompt && (
                        <details className="border-t border-neutral-700/30">
                          <summary className="px-3 py-2 text-xs text-purple-400 cursor-pointer hover:text-purple-300 hover:bg-neutral-800/30 transition-colors">
                            Show editing instructions
                          </summary>
                          <div className="px-3 py-2 bg-neutral-800/50 text-xs text-neutral-300 border-t border-neutral-700/30">
                            {iter.editing_prompt}
                          </div>
                        </details>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {/* Prompt Adapter Specific - Show Prompt Flow */}
        {node.id === 'prompt_adapter' && (data.original_prompt || data.adapted_prompt) && (
          <div>
            <h3 className="text-sm font-medium text-text-secondary mb-2">Prompt Transformation</h3>
            <div className="space-y-3">
              {data.original_prompt && (
                <div>
                  <div className="text-xs text-neutral-500 mb-1">Original Prompt:</div>
                  <div className="px-3 py-2 bg-surface-dark border border-border-base rounded">
                    <p className="text-sm text-text-primary">{data.original_prompt}</p>
                  </div>
                </div>
              )}

              {data.issues_addressed !== undefined && data.issues_addressed > 0 && (
                <div className="flex items-center gap-2 text-xs text-yellow-400">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v2H7a1 1 0 100 2h2v2a1 1 0 102 0v-2h2a1 1 0 100-2h-2V7z" />
                  </svg>
                  <span>{data.issues_addressed} issues addressed</span>
                </div>
              )}

              {data.adapted_prompt && (
                <div>
                  <div className="text-xs text-neutral-500 mb-1">Adapted for {data.model}:</div>
                  <div className="px-3 py-2 bg-green-900/20 border border-green-500/50 rounded">
                    <p className="text-sm text-green-100">{data.adapted_prompt}</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* CLIP RAG Search Specific */}
        {node.id === 'clip_rag_search' && data.search_results && (
          <div>
            <h3 className="text-sm font-medium text-text-secondary mb-2">
              🖼️ Similar Images Found ({data.search_results.length})
            </h3>
            <div className="space-y-3 max-h-[500px] overflow-y-auto">
              {data.search_results.slice(0, 10).map((result: any, index: number) => {
                // Parse the image path to build API URL
                const imagePath = result.path || result.image_path || '';
                const imageUrl = imagePath ? `http://localhost:8000/api/images/serve-by-path?path=${encodeURIComponent(imagePath)}` : null;

                return (
                  <div
                    key={index}
                    className="p-3 bg-surface-dark border border-border-base rounded hover:border-blue-500/50 transition-colors"
                  >
                    {/* Header with rank and similarity */}
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-mono text-text-tertiary">#{index + 1}</span>
                      <span className={`text-xs font-mono font-medium ${
                        result.score >= 0.8 ? 'text-green-400' :
                        result.score >= 0.6 ? 'text-yellow-400' : 'text-orange-400'
                      }`}>
                        Similarity: {(result.score * 100).toFixed(1)}%
                      </span>
                    </div>

                    {/* Image Display */}
                    {imageUrl && (
                      <div className="mb-2 rounded overflow-hidden bg-neutral-800">
                        <img
                          src={imageUrl}
                          alt={`Similar image ${index + 1}`}
                          className="w-full h-auto object-contain max-h-48"
                          onError={(e) => {
                            // Fallback to path text if image fails to load
                            e.currentTarget.style.display = 'none';
                            const parent = e.currentTarget.parentElement;
                            if (parent) {
                              parent.innerHTML = `<div class="p-2 text-xs text-red-400">Failed to load image</div><div class="p-2 text-xs text-neutral-400 break-all">${imagePath}</div>`;
                            }
                          }}
                        />
                      </div>
                    )}

                    {/* Metadata */}
                    {result.metadata && (
                      <div className="text-xs text-text-tertiary space-y-1">
                        {result.metadata.category && (
                          <div className="flex items-center gap-1">
                            <span>📁</span>
                            <span className="text-blue-300">{result.metadata.category}</span>
                          </div>
                        )}
                        {result.metadata.caption && (
                          <div className="text-neutral-400 italic line-clamp-2">
                            "{result.metadata.caption}"
                          </div>
                        )}
                      </div>
                    )}

                    {/* File path (collapsed) */}
                    <details className="mt-2">
                      <summary className="text-xs text-neutral-500 cursor-pointer hover:text-neutral-400">
                        Show file path
                      </summary>
                      <div className="mt-1 text-xs text-neutral-600 break-all font-mono">
                        {imagePath}
                      </div>
                    </details>
                  </div>
                );
              })}
              {data.search_results.length > 10 && (
                <div className="text-xs text-center text-neutral-500 py-2">
                  ... and {data.search_results.length - 10} more results
                </div>
              )}
            </div>
          </div>
        )}

        {/* Text KB Query Specific */}
        {node.id === 'text_kb_query' && data.kb_results && (
          <div>
            <h3 className="text-sm font-medium text-text-secondary mb-2">
              📚 Knowledge Base Results ({data.kb_results.length})
            </h3>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {data.kb_results.slice(0, 5).map((result: any, index: number) => (
                <div
                  key={index}
                  className="px-3 py-2 bg-surface-dark border border-border-base rounded"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-mono text-text-tertiary">#{index + 1}</span>
                    <span className={`text-xs font-mono font-medium ${
                      result.score >= 0.8 ? 'text-green-400' :
                      result.score >= 0.6 ? 'text-yellow-400' : 'text-orange-400'
                    }`}>
                      Relevance: {(result.score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="text-sm text-text-primary">
                    {result.text || result.content}
                  </div>
                </div>
              ))}
              {data.kb_results.length > 5 && (
                <div className="text-xs text-center text-neutral-500 py-2">
                  ... and {data.kb_results.length - 5} more results
                </div>
              )}
            </div>
          </div>
        )}

        {/* Reference Selector Specific */}
        {node.id === 'reference_selector' && data.selected && (
          <div>
            <h3 className="text-sm font-medium text-text-secondary mb-2">✓ Selected Reference</h3>
            <div className="p-3 bg-green-900/20 border border-green-500/50 rounded space-y-2">
              {/* Score and filename */}
              <div className="flex items-center justify-between">
                <span className="text-sm text-green-100">{data.selected}</span>
                <span className="text-sm font-mono font-medium text-green-400">
                  Score: {data.score?.toFixed(3)}
                </span>
              </div>

              {/* Show the actual image if path is available */}
              {data.selected_path && (
                <div className="rounded overflow-hidden bg-neutral-800">
                  <img
                    src={`http://localhost:8000/api/images/serve-by-path?path=${encodeURIComponent(data.selected_path)}`}
                    alt="Selected reference"
                    className="w-full h-auto object-contain max-h-64"
                    onError={(e) => {
                      e.currentTarget.style.display = 'none';
                    }}
                  />
                </div>
              )}
            </div>
          </div>
        )}

        {/* Raw Data */}
        <div>
          <h3 className="text-sm font-medium text-text-secondary mb-2">Raw Data</h3>
          <div className="px-3 py-2 bg-surface-dark border border-border-base rounded">
            <pre className="text-xs text-text-tertiary font-mono overflow-x-auto">
              {JSON.stringify(data, null, 2)}
            </pre>
          </div>
        </div>
      </div>
    </div>
  )
}

function getStatusColor(status: string): string {
  switch (status) {
    case 'completed':
      return 'text-white'
    case 'processing':
      return 'text-white animate-pulse'
    case 'error':
      return 'text-status-error'
    default:
      return 'text-status-pending'
  }
}
