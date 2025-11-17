'use client'

import { useState } from 'react'
import { JobProposal } from '@/lib/types'
import { X, CheckCircle, XCircle, AlertCircle } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface JobApprovalPanelProps {
  proposals: JobProposal[]
  pipelineId: string
  onApprove: (proposalIds: string[]) => Promise<void>
  onReject: (proposalIds: string[]) => Promise<void>
  onClose: () => void
}

export default function JobApprovalPanel({
  proposals,
  pipelineId,
  onApprove,
  onReject,
  onClose,
}: JobApprovalPanelProps) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set(proposals.map(p => p.id)))
  const [loading, setLoading] = useState(false)

  const toggleSelection = (id: string) => {
    const newSelected = new Set(selectedIds)
    if (newSelected.has(id)) {
      newSelected.delete(id)
    } else {
      newSelected.add(id)
    }
    setSelectedIds(newSelected)
  }

  const handleApprove = async () => {
    if (selectedIds.size === 0) {
      alert('Please select at least one proposal')
      return
    }

    setLoading(true)
    try {
      await onApprove(Array.from(selectedIds))
    } finally {
      setLoading(false)
    }
  }

  const handleReject = async () => {
    const idsToReject = proposals.map(p => p.id).filter(id => !selectedIds.has(id))

    if (idsToReject.length === 0) {
      // Just close if no rejections
      onClose()
      return
    }

    setLoading(true)
    try {
      await onReject(idsToReject)
      if (selectedIds.size > 0) {
        await onApprove(Array.from(selectedIds))
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/70 z-40 animate-fade-in backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] bg-gradient-to-b from-neutral-900 to-neutral-950 border-2 border-yellow-500/30 rounded-xl z-50 animate-scale-in max-h-[85vh] flex flex-col shadow-2xl shadow-yellow-500/10">
        {/* Header */}
        <div className="px-6 py-4 border-b border-neutral-800/50 flex items-center justify-between bg-neutral-900/50 backdrop-blur-sm rounded-t-xl">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-lg bg-yellow-500/10 border border-yellow-500/30 flex items-center justify-center">
                <AlertCircle className="w-5 h-5 text-yellow-400" />
              </div>
              <div className="absolute -top-0.5 -right-0.5 w-3 h-3 bg-yellow-500 rounded-full animate-pulse" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">
                Data Collection Proposals
              </h2>
              <p className="text-sm text-neutral-400 mt-0.5">
                {proposals.length} {proposals.length === 1 ? 'gap' : 'gaps'} identified • Review before creating jobs
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-neutral-800 rounded-lg transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5 text-neutral-400 hover:text-white" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="space-y-4">
            {proposals.map((proposal, index) => (
              <div
                key={proposal.id}
                onClick={() => toggleSelection(proposal.id)}
                className={`
                  group relative rounded-xl border transition-all duration-200 cursor-pointer overflow-hidden
                  ${selectedIds.has(proposal.id)
                    ? 'border-green-500/50 bg-gradient-to-br from-green-900/20 to-green-950/30 shadow-lg shadow-green-500/10'
                    : 'border-neutral-700/50 bg-neutral-800/40 hover:border-neutral-600 hover:bg-neutral-800/60'
                  }
                `}
              >
                {/* Selection Indicator Bar */}
                {selectedIds.has(proposal.id) && (
                  <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-green-400 to-green-600" />
                )}

                <div className="p-5 pl-6">
                  <div className="flex items-start gap-4">
                    {/* Checkbox */}
                    <div className="mt-0.5 flex-shrink-0">
                      <div className={`
                        w-6 h-6 rounded-md border-2 flex items-center justify-center transition-all duration-200
                        ${selectedIds.has(proposal.id)
                          ? 'border-green-500 bg-green-500 scale-110'
                          : 'border-neutral-600 bg-neutral-900/50 group-hover:border-neutral-500'
                        }
                      `}>
                        {selectedIds.has(proposal.id) && (
                          <CheckCircle className="w-4 h-4 text-white" />
                        )}
                      </div>
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      {/* Header: Category, Subcategory, Priority */}
                      <div className="flex items-center gap-2 mb-3 flex-wrap">
                        <span className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-blue-500/10 border border-blue-500/30 rounded-full text-xs font-medium text-blue-300">
                          <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                          {proposal.category}
                        </span>
                        {proposal.subcategory && (
                          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-purple-500/10 border border-purple-500/30 rounded-full text-xs font-medium text-purple-300">
                            <span className="w-1.5 h-1.5 rounded-full bg-purple-400" />
                            {proposal.subcategory}
                          </span>
                        )}
                        <span className={`
                          inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold ml-auto
                          ${proposal.priority >= 4 ? 'bg-red-500/20 border border-red-500/50 text-red-300' :
                            proposal.priority === 3 ? 'bg-orange-500/20 border border-orange-500/50 text-orange-300' :
                            proposal.priority === 2 ? 'bg-yellow-500/20 border border-yellow-500/50 text-yellow-300' :
                            'bg-neutral-600/20 border border-neutral-500/50 text-neutral-300'
                          }
                        `}>
                          <AlertCircle className="w-3.5 h-3.5" />
                          P{proposal.priority}
                        </span>
                      </div>

                      {/* Reason with Markdown */}
                      <div className="prose prose-invert prose-sm max-w-none mb-3">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            p: ({ children }) => (
                              <p className="text-sm text-neutral-300 leading-relaxed mb-2 last:mb-0">
                                {children}
                              </p>
                            ),
                            strong: ({ children }) => (
                              <strong className="text-white font-semibold">{children}</strong>
                            ),
                            ul: ({ children }) => (
                              <ul className="list-none space-y-1.5 text-sm text-neutral-300 ml-0 my-2">
                                {children}
                              </ul>
                            ),
                            ol: ({ children }) => (
                              <ol className="list-decimal list-inside space-y-1.5 text-sm text-neutral-300 ml-2 my-2">
                                {children}
                              </ol>
                            ),
                            li: ({ children }) => (
                              <li className="text-sm text-neutral-300 flex items-start gap-2 before:content-['•'] before:text-yellow-500 before:font-bold before:text-base">
                                <span className="flex-1">{children}</span>
                              </li>
                            ),
                            h1: ({ children }) => (
                              <h1 className="text-base font-bold text-white mb-2 mt-3 first:mt-0">
                                {children}
                              </h1>
                            ),
                            h2: ({ children }) => (
                              <h2 className="text-sm font-bold text-white mb-2 mt-2 first:mt-0">
                                {children}
                              </h2>
                            ),
                            h3: ({ children }) => (
                              <h3 className="text-sm font-semibold text-neutral-200 mb-1 mt-2 first:mt-0">
                                {children}
                              </h3>
                            ),
                            code: ({ children }) => (
                              <code className="px-1.5 py-0.5 bg-neutral-700/50 border border-neutral-600/50 rounded text-xs text-green-300 font-mono">
                                {children}
                              </code>
                            ),
                            blockquote: ({ children }) => (
                              <blockquote className="border-l-2 border-yellow-500/50 pl-3 italic text-neutral-400">
                                {children}
                              </blockquote>
                            ),
                          }}
                        >
                          {proposal.reason}
                        </ReactMarkdown>
                      </div>

                      {/* Keywords */}
                      {proposal.keywords && proposal.keywords.length > 0 && (
                        <div className="flex flex-wrap gap-1.5 pt-2 border-t border-neutral-700/30">
                          {proposal.keywords.slice(0, 8).map((keyword, idx) => (
                            <span
                              key={idx}
                              className="px-2.5 py-1 bg-neutral-700/30 border border-neutral-600/30 rounded-md text-xs text-neutral-400 font-medium hover:bg-neutral-700/50 hover:text-neutral-300 transition-colors"
                            >
                              {keyword}
                            </span>
                          ))}
                          {proposal.keywords.length > 8 && (
                            <span className="px-2.5 py-1 text-xs text-neutral-500 font-medium">
                              +{proposal.keywords.length - 8} more
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer Actions */}
        <div className="px-6 py-4 border-t border-neutral-800/50 bg-neutral-950/80 backdrop-blur-sm flex items-center justify-between rounded-b-xl">
          <div className="text-sm">
            <span className="text-neutral-400">Selected: </span>
            <span className="text-white font-semibold">{selectedIds.size}</span>
            <span className="text-neutral-500"> / {proposals.length}</span>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => onReject(proposals.map(p => p.id))}
              disabled={loading}
              className="px-4 py-2.5 bg-neutral-800 border border-neutral-700 text-neutral-300 rounded-lg hover:bg-neutral-700 hover:text-white hover:border-neutral-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 font-medium"
            >
              <XCircle className="w-4 h-4" />
              Reject All
            </button>
            <button
              onClick={handleReject}
              disabled={loading || selectedIds.size === 0}
              className="px-4 py-2.5 bg-neutral-800 border border-neutral-700 text-neutral-300 rounded-lg hover:bg-neutral-700 hover:text-white hover:border-neutral-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed font-medium"
            >
              Approve & Reject Others
            </button>
            <button
              onClick={handleApprove}
              disabled={loading || selectedIds.size === 0}
              className="px-6 py-2.5 bg-gradient-to-r from-green-600 to-green-500 text-white font-semibold rounded-lg hover:from-green-500 hover:to-green-400 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shadow-lg shadow-green-500/20"
            >
              <CheckCircle className="w-4 h-4" />
              {loading ? 'Creating...' : `Approve (${selectedIds.size})`}
            </button>
          </div>
        </div>
      </div>
    </>
  )
}
