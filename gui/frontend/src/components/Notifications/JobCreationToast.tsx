"use client";

import { useEffect, useState } from "react";
import { JobCreationMessage } from "@/lib/types";

interface JobCreationToastProps {
  notifications: JobCreationMessage[];
  onDismiss: (index: number) => void;
}

export default function JobCreationToast({
  notifications,
  onDismiss,
}: JobCreationToastProps) {
  if (notifications.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 space-y-2 max-w-md">
      {notifications.map((notification, index) => (
        <div
          key={index}
          className={`bg-neutral-900 border-2 rounded-xl p-4 shadow-2xl animate-slide-in-right ${
            notification.status === "created"
              ? "border-green-500"
              : notification.status === "creating"
              ? "border-yellow-500"
              : notification.status === "error"
              ? "border-red-500"
              : notification.status === "skipped"
              ? "border-gray-500"
              : "border-blue-500"
          }`}
        >
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1">
              {/* Icon */}
              <div className="flex items-center gap-2 mb-2">
                {notification.status === "created" && (
                  <span className="text-green-400 text-lg">✓</span>
                )}
                {notification.status === "creating" && (
                  <span className="text-yellow-400 text-lg">⏳</span>
                )}
                {notification.status === "error" && (
                  <span className="text-red-400 text-lg">✗</span>
                )}
                {notification.status === "skipped" && (
                  <span className="text-gray-400 text-lg">⊘</span>
                )}
                {notification.status === "no_gaps" && (
                  <span className="text-blue-400 text-lg">ℹ</span>
                )}
                <span className="font-medium text-sm text-neutral-200">
                  {notification.status === "created" && "Job Created"}
                  {notification.status === "creating" && "Creating Job"}
                  {notification.status === "error" && "Job Creation Failed"}
                  {notification.status === "skipped" && "Job Skipped"}
                  {notification.status === "no_gaps" && "No Data Gaps"}
                </span>
              </div>

              {/* Message */}
              <p className="text-sm text-neutral-300">{notification.message}</p>

              {/* Details */}
              {notification.gap && (
                <div className="mt-2 text-xs text-neutral-500 space-y-1">
                  <div>
                    <span className="font-mono">
                      {notification.gap.category}
                      {notification.gap.subcategory &&
                        ` / ${notification.gap.subcategory}`}
                    </span>
                  </div>
                  {notification.gap.keywords &&
                    notification.gap.keywords.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {notification.gap.keywords.slice(0, 5).map((keyword, i) => (
                          <span
                            key={i}
                            className="px-1.5 py-0.5 bg-neutral-800 rounded text-xs"
                          >
                            {keyword}
                          </span>
                        ))}
                      </div>
                    )}
                </div>
              )}

              {/* Job ID */}
              {notification.job_id && (
                <div className="mt-2 text-xs font-mono text-green-400">
                  Job ID: {notification.job_id}
                </div>
              )}

              {/* Error */}
              {notification.error && (
                <div className="mt-2 text-xs text-red-400">
                  Error: {notification.error}
                </div>
              )}
            </div>

            {/* Close button */}
            <button
              onClick={() => onDismiss(index)}
              className="text-neutral-500 hover:text-neutral-300 transition-colors"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
