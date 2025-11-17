"use client";

import { useState, useEffect } from "react";
import { historyAPI } from "@/lib/api";
import type { HistorySummary } from "@/lib/types";

interface HistoryPanelProps {
  onSelectHistory: (pipelineId: string) => void;
  isVisible: boolean;
  onToggle: () => void;
}

export default function HistoryPanel({
  onSelectHistory,
  isVisible,
  onToggle,
}: HistoryPanelProps) {
  const [history, setHistory] = useState<HistorySummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load history on mount and when visible
  useEffect(() => {
    if (isVisible) {
      loadHistory();
    }
  }, [isVisible]);

  const loadHistory = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await historyAPI.list(50);
      setHistory(data);
    } catch (err: any) {
      console.error("Failed to load history:", err);
      setError(err.message || "Failed to load history");
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 60) {
      return `${diffMins}m ago`;
    } else if (diffHours < 24) {
      return `${diffHours}h ago`;
    } else if (diffDays < 7) {
      return `${diffDays}d ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  const formatDuration = (duration?: number) => {
    if (!duration) return "N/A";
    const mins = Math.floor(duration / 60);
    const secs = Math.floor(duration % 60);
    if (mins > 0) {
      return `${mins}m ${secs}s`;
    }
    return `${secs}s`;
  };

  if (!isVisible) {
    // Collapsed state - show toggle button
    return (
      <div className="fixed left-0 top-1/2 -translate-y-1/2 z-50">
        <button
          onClick={onToggle}
          className="bg-neutral-900 border border-neutral-700 border-l-0 rounded-r-lg p-2 hover:bg-neutral-800 transition-colors"
          title="Show History"
        >
          <svg
            className="w-5 h-5 text-neutral-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5l7 7-7 7"
            />
          </svg>
        </button>
      </div>
    );
  }

  return (
    <div className="fixed left-0 top-0 h-full w-80 bg-neutral-900 border-r border-neutral-700 z-50 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-neutral-700">
        <h2 className="text-sm font-medium text-neutral-200">History</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={loadHistory}
            className="p-1 hover:bg-neutral-800 rounded transition-colors"
            title="Refresh"
          >
            <svg
              className="w-4 h-4 text-neutral-400"
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
          </button>
          <button
            onClick={onToggle}
            className="p-1 hover:bg-neutral-800 rounded transition-colors"
            title="Hide History"
          >
            <svg
              className="w-4 h-4 text-neutral-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 19l-7-7 7-7"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {loading && (
          <div className="flex items-center justify-center h-32">
            <div className="text-sm text-neutral-500">Loading...</div>
          </div>
        )}

        {error && (
          <div className="p-4">
            <div className="bg-red-900/20 border border-red-500/50 rounded p-3 text-sm text-red-400">
              {error}
            </div>
          </div>
        )}

        {!loading && !error && history.length === 0 && (
          <div className="flex flex-col items-center justify-center h-32 text-neutral-500">
            <svg
              className="w-12 h-12 mb-2 opacity-50"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <p className="text-sm">No history yet</p>
          </div>
        )}

        {!loading && !error && history.length > 0 && (
          <div className="divide-y divide-neutral-800">
            {history.map((item) => (
              <button
                key={item.pipeline_id}
                onClick={() => onSelectHistory(item.pipeline_id)}
                className="w-full text-left px-4 py-3 hover:bg-neutral-800 transition-colors"
              >
                {/* Timestamp & Status */}
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-neutral-500">
                    {formatDate(item.timestamp)}
                  </span>
                  <span
                    className={`text-xs px-1.5 py-0.5 rounded ${
                      item.status === "completed"
                        ? "bg-green-900/30 text-green-400"
                        : item.status === "error"
                        ? "bg-red-900/30 text-red-400"
                        : "bg-yellow-900/30 text-yellow-400"
                    }`}
                  >
                    {item.status}
                  </span>
                </div>

                {/* Prompt */}
                <p className="text-sm text-neutral-200 mb-2 line-clamp-2">
                  {item.prompt}
                </p>

                {/* Details */}
                <div className="flex items-center gap-3 text-xs text-neutral-500">
                  <span title="Country">{item.country}</span>
                  {item.iterations > 0 && (
                    <span title="Iterations">{item.iterations} iter</span>
                  )}
                  {item.final_score != null && (
                    <span
                      title="Final Cultural Score"
                      className="text-neutral-400"
                    >
                      {item.final_score.toFixed(1)}/10
                    </span>
                  )}
                  {item.duration && (
                    <span title="Duration">{formatDuration(item.duration)}</span>
                  )}
                </div>

                {/* Models */}
                <div className="mt-1 text-xs text-neutral-600 font-mono">
                  {item.t2i_model} → {item.i2i_model}
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
