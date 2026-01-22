"use client";

import { useState } from "react";
import type { PipelineConfig } from "@/lib/types";

interface ConfigSidebarProps {
  onStart: (config: PipelineConfig, imageFile: File | null) => void;
  isVisible: boolean;
  onToggle: () => void;
}

export default function ConfigSidebar({
  onStart,
  isVisible,
  onToggle,
}: ConfigSidebarProps) {
  const [config, setConfig] = useState<PipelineConfig>({
    prompt: "A Korean woman in traditional hanbok",
    country: "korea",
    category: "traditional_clothing",
    t2i_model: "sd35",
    i2i_model: "qwen",
    max_iterations: 3,
    target_score: 8.0,
    load_in_4bit: true,
  });
  const [imageFile, setImageFile] = useState<File | null>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setImageFile(e.target.files[0]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onStart(config, imageFile);
  };

  if (!isVisible) {
    return (
      <div className="fixed right-0 top-1/2 -translate-y-1/2 z-30">
        <button
          onClick={onToggle}
          className="bg-neutral-900 border border-neutral-700 border-r-0 rounded-l-lg p-3 hover:bg-neutral-800 transition-colors shadow-lg"
          title="Show Configuration"
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
              d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
        </button>
      </div>
    );
  }

  return (
    <div className="fixed right-0 top-0 h-full w-96 bg-neutral-900 border-l border-neutral-700 z-30 flex flex-col shadow-2xl animate-slide-in">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-neutral-800 bg-neutral-950">
        <div>
          <h2 className="text-lg font-medium text-neutral-200">Configuration</h2>
          <p className="text-xs text-neutral-500 mt-0.5">
            Configure pipeline settings
          </p>
        </div>
        <button
          onClick={onToggle}
          className="p-1 hover:bg-neutral-800 rounded transition-colors"
          title="Hide Configuration"
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
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Prompt */}
          <div>
            <label className="block text-sm font-medium text-neutral-300 mb-1">
              Prompt
            </label>
            <textarea
              className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded text-sm text-neutral-200 placeholder-neutral-500 focus:outline-none focus:border-neutral-500 transition-colors"
              rows={3}
              placeholder="Enter your prompt..."
              value={config.prompt}
              onChange={(e) => setConfig({ ...config, prompt: e.target.value })}
              required
            />
          </div>

          {/* Image Upload */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-neutral-300">Or Upload an Image</label>
            <div className="flex items-center space-x-2">
              <input
                type="file"
                onChange={handleImageChange}
                className="block w-full text-sm text-neutral-400 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-neutral-800 file:text-neutral-200 hover:file:bg-neutral-700"
                accept="image/*"
              />
              {imageFile && (
                <button
                  onClick={() => setImageFile(null)}
                  className="px-2 py-1 text-xs text-neutral-400 hover:text-white"
                >
                  Clear
                </button>
              )}
            </div>
            {imageFile && <p className="text-xs text-neutral-400">Selected: {imageFile.name}</p>}
          </div>

          {/* Country & Category */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-neutral-300 mb-1">
                Country
              </label>
              <select
                className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded text-sm text-neutral-200 focus:outline-none focus:border-neutral-500 transition-colors"
                value={config.country}
                onChange={(e) => setConfig({ ...config, country: e.target.value })}
              >
                <option value="korea">🇰🇷 Korea (328)</option>
                <option value="china">🇨🇳 China (202)</option>
                <option value="japan">🇯🇵 Japan (200)</option>
                <option value="usa">🇺🇸 USA (136)</option>
                <option value="nigeria">🇳🇬 Nigeria (115)</option>
                <option value="mexico">🇲🇽 Mexico (27)</option>
                <option value="kenya">🇰🇪 Kenya (24)</option>
                <option value="italy">🇮🇹 Italy (23)</option>
                <option value="france">🇫🇷 France (17)</option>
                <option value="germany">🇩🇪 Germany (16)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-neutral-300 mb-1">
                Category
              </label>
              <select
                className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded text-sm text-neutral-200 focus:outline-none focus:border-neutral-500 transition-colors"
                value={config.category}
                onChange={(e) => setConfig({ ...config, category: e.target.value })}
              >
                <option value="traditional_clothing">Clothing</option>
                <option value="food">Food</option>
                <option value="architecture">Architecture</option>
                <option value="festivals">Festivals</option>
                <option value="traditional_arts">Arts</option>
                <option value="religious_sites">Religious Sites</option>
                <option value="handicrafts">Handicrafts</option>
                <option value="music_and_dance">Music & Dance</option>
                <option value="sports">Sports</option>
                <option value="ceremonies">Ceremonies</option>
                <option value="daily_life">Daily Life</option>
                <option value="nature">Nature</option>
              </select>
            </div>
          </div>

          {/* Models */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-neutral-300 mb-1">
                T2I Model
              </label>
              <select
                className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded text-sm text-neutral-200 focus:outline-none focus:border-neutral-500 transition-colors"
                value={config.t2i_model}
                onChange={(e) => setConfig({ ...config, t2i_model: e.target.value })}
                disabled={!!imageFile}
              >
                <option value="sd35">SD 3.5 Medium</option>
                <option value="flux">FLUX.1-dev</option>
                <option value="sdxl">SDXL</option>
                <option value="gemini">Gemini Imagen 3</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-neutral-300 mb-1">
                I2I Model
              </label>
              <select
                className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded text-sm text-neutral-200 focus:outline-none focus:border-neutral-500 transition-colors"
                value={config.i2i_model}
                onChange={(e) => setConfig({ ...config, i2i_model: e.target.value })}
              >
                <option value="qwen">Qwen-Image-Edit</option>
                <option value="flux">FLUX Kontext</option>
                <option value="sdxl">SDXL InstructPix2Pix</option>
                <option value="sd35">SD3.5 Inpainting</option>
              </select>
            </div>
          </div>

          {/* Advanced Settings */}
          <div className="space-y-3 pt-2 border-t border-neutral-800">
            <div>
              <label className="block text-sm font-medium text-neutral-300 mb-1">
                Max Iterations: {config.max_iterations}
              </label>
              <input
                type="range"
                min="1"
                max="5"
                className="w-full"
                value={config.max_iterations}
                onChange={(e) =>
                  setConfig({ ...config, max_iterations: parseInt(e.target.value) })
                }
                style={{
                  '--range-progress': `${((config.max_iterations - 1) / (5 - 1)) * 100}%`
                } as React.CSSProperties}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-neutral-300 mb-1">
                Target Score: {config.target_score.toFixed(1)}
              </label>
              <input
                type="range"
                min="7.0"
                max="10.0"
                step="0.5"
                className="w-full"
                value={config.target_score}
                onChange={(e) =>
                  setConfig({ ...config, target_score: parseFloat(e.target.value) })
                }
                style={{
                  '--range-progress': `${((config.target_score - 7.0) / (10.0 - 7.0)) * 100}%`
                } as React.CSSProperties}
              />
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                id="load_in_4bit"
                className="w-4 h-4 bg-neutral-800 border border-neutral-700 rounded"
                checked={config.load_in_4bit}
                onChange={(e) =>
                  setConfig({ ...config, load_in_4bit: e.target.checked })
                }
              />
              <label
                htmlFor="load_in_4bit"
                className="ml-2 text-sm text-neutral-400"
              >
                Use 4-bit quantization (saves VRAM)
              </label>
            </div>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            className="w-full px-4 py-3 bg-white text-neutral-900 font-medium rounded-lg hover:bg-neutral-200 transition-colors"
          >
            🚀 Start Pipeline
          </button>
        </form>
      </div>
    </div>
  );
}
