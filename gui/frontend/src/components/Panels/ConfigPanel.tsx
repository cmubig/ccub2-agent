'use client'

import React, { useState } from 'react'
import { PipelineConfig } from '@/lib/types'

interface ConfigPanelProps {
  onStart: (config: PipelineConfig, imageFile: File | null) => void
}

export default function ConfigPanel({ onStart }: ConfigPanelProps) {
  const [config, setConfig] = useState<PipelineConfig>({
    prompt: 'A Korean woman in traditional hanbok',
    country: 'korea',
    category: 'traditional_clothing',
    t2i_model: 'sdxl',
    i2i_model: 'qwen',
    max_iterations: 3,
    target_score: 8.0,
    load_in_4bit: true,
  })
  const [imageFile, setImageFile] = useState<File | null>(null)

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setImageFile(e.target.files[0])
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onStart(config, imageFile)
  }

  return (
    <div className="bg-surface-elevated border border-border-base rounded-lg shadow-panel overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border-dark">
        <h2 className="text-lg font-medium text-text-primary">Pipeline Configuration</h2>
        <p className="text-xs text-text-tertiary mt-1">Configure and start the CCUB2 pipeline</p>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="p-4 space-y-4">
        {/* Prompt */}
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1">Prompt</label>
          <textarea
            value={config.prompt}
            onChange={(e) => setConfig({ ...config, prompt: e.target.value })}
            className="w-full px-3 py-2 bg-surface-dark border border-border-base rounded text-sm text-text-primary placeholder-text-tertiary focus:outline-none focus:border-border-accent transition-colors"
            rows={3}
            placeholder="Enter your prompt..."
            required
          />
        </div>

        {/* Image Upload */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-text-secondary">Or Upload an Image</label>
          <div className="flex items-center space-x-2">
            <input
              type="file"
              onChange={handleImageChange}
              className="block w-full text-sm text-text-tertiary file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-surface-dark file:text-text-primary hover:file:bg-surface-hover"
              accept="image/*"
            />
            {imageFile && (
              <button
                onClick={() => setImageFile(null)}
                className="px-2 py-1 text-xs text-text-tertiary"
              >
                Clear
              </button>
            )}
          </div>
          {imageFile && <p className="text-xs text-text-tertiary">Selected: {imageFile.name}</p>}
        </div>

        {/* Country & Category */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1">Country</label>
            <select
              value={config.country}
              onChange={(e) => setConfig({ ...config, country: e.target.value })}
              className="w-full px-3 py-2 bg-surface-dark border border-border-base rounded text-sm text-text-primary focus:outline-none focus:border-border-accent transition-colors"
            >
              <option value="korea">Korea</option>
              <option value="japan">Japan</option>
              <option value="china">China</option>
              <option value="india">India</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1">Category</label>
            <select
              value={config.category}
              onChange={(e) => setConfig({ ...config, category: e.target.value })}
              className="w-full px-3 py-2 bg-surface-dark border border-border-base rounded text-sm text-text-primary focus:outline-none focus:border-border-accent transition-colors"
            >
              <option value="traditional_clothing">Traditional Clothing</option>
              <option value="food">Food</option>
              <option value="architecture">Architecture</option>
              <option value="festivals">Festivals</option>
            </select>
          </div>
        </div>

        {/* Models */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1">T2I Model</label>
            <select
              value={config.t2i_model}
              onChange={(e) => setConfig({ ...config, t2i_model: e.target.value })}
              className="w-full px-3 py-2 bg-surface-dark border border-border-base rounded text-sm text-text-primary focus:outline-none focus:border-border-accent transition-colors"
              disabled={!!imageFile}
            >
              <option value="sdxl">SDXL</option>
              <option value="flux">FLUX.1-dev</option>
              <option value="sd35">SD 3.5 Medium</option>
              <option value="gemini">Gemini Imagen 3</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1">I2I Model</label>
            <select
              value={config.i2i_model}
              onChange={(e) => setConfig({ ...config, i2i_model: e.target.value })}
              className="w-full px-3 py-2 bg-surface-dark border border-border-base rounded text-sm text-text-primary focus:outline-none focus:border-border-accent transition-colors"
            >
              <option value="qwen">Qwen-Image-Edit</option>
              <option value="flux">FLUX Kontext</option>
              <option value="sdxl">SDXL InstructPix2Pix</option>
              <option value="sd35">SD3.5 Inpainting</option>
            </select>
          </div>
        </div>

        {/* Advanced Settings */}
        <div className="space-y-3 pt-2 border-t border-border-dark">
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1">
              Max Iterations: {config.max_iterations}
            </label>
            <input
              type="range"
              min="1"
              max="5"
              value={config.max_iterations}
              onChange={(e) => setConfig({ ...config, max_iterations: parseInt(e.target.value) })}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1">
              Target Score: {config.target_score.toFixed(1)}
            </label>
            <input
              type="range"
              min="7.0"
              max="10.0"
              step="0.5"
              value={config.target_score}
              onChange={(e) => setConfig({ ...config, target_score: parseFloat(e.target.value) })}
              className="w-full"
            />
          </div>

          <div className="flex items-center">
            <input
              type="checkbox"
              id="load_in_4bit"
              checked={config.load_in_4bit}
              onChange={(e) => setConfig({ ...config, load_in_4bit: e.target.checked })}
              className="w-4 h-4 bg-surface-dark border border-border-base rounded"
            />
            <label htmlFor="load_in_4bit" className="ml-2 text-sm text-text-secondary">
              Use 4-bit quantization (saves VRAM)
            </label>
          </div>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          className="w-full px-4 py-3 bg-white text-background-primary font-medium rounded hover:bg-text-secondary transition-colors"
        >
          🚀 Start Pipeline
        </button>
      </form>
    </div>
  )
}
