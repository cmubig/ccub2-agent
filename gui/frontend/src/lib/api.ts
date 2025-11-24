/**
 * API Client for CCUB2 Backend
 */

import axios from 'axios'
import { PipelineConfig, PipelineState, BaseNodeData } from './types'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const pipelineAPI = {
  /**
   * Start pipeline execution
   */
  start: async (config: PipelineConfig) => {
    const response = await api.post('/api/pipeline/start', { config })
    return response.data
  },

  /**
   * Get pipeline status
   */
  getStatus: async (): Promise<PipelineState> => {
    const response = await api.get('/api/pipeline/status')
    return response.data
  },

  /**
   * Stop pipeline execution
   */
  stop: async () => {
    const response = await api.post('/api/pipeline/stop')
    return response.data
  },

  /**
   * Get available countries
   */
  getCountries: async () => {
    const response = await api.get('/api/pipeline/countries')
    return response.data
  },

  /**
   * Get available models
   */
  getModels: async () => {
    const response = await api.get('/api/pipeline/models')
    return response.data
  },
}

export const nodesAPI = {
  /**
   * Get node details
   */
  getNode: async (nodeId: string): Promise<BaseNodeData> => {
    const response = await api.get(`/api/nodes/${nodeId}`)
    return response.data
  },

  /**
   * Get all nodes
   */
  getAllNodes: async () => {
    const response = await api.get('/api/nodes')
    return response.data
  },

  /**
   * Get node outputs
   */
  getNodeOutputs: async (nodeId: string) => {
    const response = await api.get(`/api/nodes/${nodeId}/outputs`)
    return response.data
  },
}

export const historyAPI = {
  /**
   * List pipeline execution history
   */
  list: async (limit: number = 50) => {
    const response = await api.get('/api/history/list', {
      params: { limit }
    })
    return response.data
  },

  /**
   * Get detailed history for a pipeline
   */
  get: async (pipelineId: string) => {
    const response = await api.get(`/api/history/${pipelineId}`)
    return response.data
  },

  /**
   * Delete a pipeline history
   */
  delete: async (pipelineId: string) => {
    const response = await api.delete(`/api/history/${pipelineId}`)
    return response.data
  },

  /**
   * Cleanup old history
   */
  cleanup: async (keepDays: number = 30) => {
    const response = await api.post('/api/history/cleanup', null, {
      params: { keep_days: keepDays }
    })
    return response.data
  },
}

export const jobsAPI = {
  /**
   * Approve job proposals
   */
  approve: async (proposalIds: string[], pipelineId: string) => {
    const response = await api.post('/api/pipeline/jobs/approve', {
      proposal_ids: proposalIds,
      pipeline_id: pipelineId
    })
    return response.data
  },

  /**
   * Reject job proposals
   */
  reject: async (proposalIds: string[], pipelineId: string) => {
    const response = await api.post('/api/pipeline/jobs/reject', {
      proposal_ids: proposalIds,
      pipeline_id: pipelineId
    })
    return response.data
  },
}

export default api
