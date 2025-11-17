"use client";

import { useEffect, useState } from "react";

interface GPUStats {
  available: boolean;
  device_name?: string;
  current_device?: number;
  gpu_count?: number;
  memory?: {
    allocated_gb: number;
    reserved_gb: number;
    total_gb: number;
    used_percent: number;
  };
  utilization_percent?: number;
  temperature_c?: number;
  error?: string;
  message?: string;
}

interface SystemStatusPanelProps {
  gpuStats: GPUStats | null;
}

export default function SystemStatusPanel({ gpuStats }: SystemStatusPanelProps) {
  // Determine GPU status color
  const getUtilizationColor = (percent?: number) => {
    if (!percent) return "text-gray-500";
    if (percent < 30) return "text-green-400";
    if (percent < 70) return "text-yellow-400";
    return "text-red-400";
  };

  const getTemperatureColor = (temp?: number) => {
    if (!temp) return "text-gray-500";
    if (temp < 60) return "text-green-400";
    if (temp < 80) return "text-yellow-400";
    return "text-red-400";
  };

  const getMemoryColor = (percent?: number) => {
    if (!percent) return "text-gray-500";
    if (percent < 50) return "text-green-400";
    if (percent < 80) return "text-yellow-400";
    return "text-red-400";
  };

  if (!gpuStats) {
    return (
      <div className="bg-neutral-900 border border-neutral-700 rounded-lg p-4">
        <h3 className="text-sm font-medium text-neutral-300 mb-3 flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-gray-500 animate-pulse"></div>
          System Status
        </h3>
        <p className="text-xs text-neutral-500">Waiting for GPU data...</p>
      </div>
    );
  }

  if (!gpuStats.available) {
    return (
      <div className="bg-neutral-900 border border-neutral-700 rounded-lg p-4">
        <h3 className="text-sm font-medium text-neutral-300 mb-3 flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-red-500"></div>
          System Status
        </h3>
        <p className="text-xs text-red-400">
          {gpuStats.error || gpuStats.message || "GPU not available"}
        </p>
      </div>
    );
  }

  const memory = gpuStats.memory;
  const utilization = gpuStats.utilization_percent;
  const temperature = gpuStats.temperature_c;

  // GPU is active if utilization > 10%
  const isActive = (utilization ?? 0) > 10;

  return (
    <div className="bg-neutral-900 border border-neutral-700 rounded-lg p-4">
      <h3 className="text-sm font-medium text-neutral-300 mb-3 flex items-center gap-2">
        <div
          className={`w-2 h-2 rounded-full ${
            isActive ? "bg-green-400 animate-pulse" : "bg-gray-500"
          }`}
        ></div>
        System Status
      </h3>

      <div className="space-y-3">
        {/* GPU Name */}
        <div>
          <p className="text-xs text-neutral-500">GPU</p>
          <p className="text-sm text-neutral-200 font-mono truncate">
            {gpuStats.device_name || "Unknown"}
          </p>
        </div>

        {/* GPU Utilization */}
        {utilization !== undefined && (
          <div>
            <div className="flex justify-between items-center mb-1">
              <p className="text-xs text-neutral-500">Utilization</p>
              <p className={`text-sm font-mono ${getUtilizationColor(utilization)}`}>
                {utilization}%
              </p>
            </div>
            <div className="w-full bg-neutral-800 rounded-full h-1.5">
              <div
                className={`h-1.5 rounded-full transition-all duration-300 ${
                  utilization < 30
                    ? "bg-green-400"
                    : utilization < 70
                    ? "bg-yellow-400"
                    : "bg-red-400"
                }`}
                style={{ width: `${utilization}%` }}
              ></div>
            </div>
          </div>
        )}

        {/* VRAM Usage */}
        {memory && (
          <div>
            <div className="flex justify-between items-center mb-1">
              <p className="text-xs text-neutral-500">VRAM</p>
              <p className={`text-sm font-mono ${getMemoryColor(memory.used_percent)}`}>
                {memory.reserved_gb.toFixed(1)} / {memory.total_gb.toFixed(1)} GB
              </p>
            </div>
            <div className="w-full bg-neutral-800 rounded-full h-1.5">
              <div
                className={`h-1.5 rounded-full transition-all duration-300 ${
                  memory.used_percent < 50
                    ? "bg-green-400"
                    : memory.used_percent < 80
                    ? "bg-yellow-400"
                    : "bg-red-400"
                }`}
                style={{ width: `${memory.used_percent}%` }}
              ></div>
            </div>
            <p className="text-xs text-neutral-500 mt-1">
              {memory.used_percent.toFixed(1)}% used
            </p>
          </div>
        )}

        {/* Temperature */}
        {temperature !== undefined && (
          <div>
            <div className="flex justify-between items-center">
              <p className="text-xs text-neutral-500">Temperature</p>
              <p className={`text-sm font-mono ${getTemperatureColor(temperature)}`}>
                {temperature}°C
              </p>
            </div>
          </div>
        )}

        {/* Device Info */}
        {gpuStats.gpu_count !== undefined && gpuStats.gpu_count > 1 && (
          <div>
            <p className="text-xs text-neutral-500">
              Device {gpuStats.current_device} of {gpuStats.gpu_count}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
