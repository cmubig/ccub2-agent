/**
 * TypeScript Type Definitions for CCUB2 Agent
 */

export type NodeStatus = 'pending' | 'processing' | 'completed' | 'error'

export type NodeType =
  | 'input'
  | 't2i_generator'
  | 'vlm_detector'
  | 'text_kb_query'
  | 'clip_rag_search'
  | 'reference_selector'
  | 'prompt_adapter'
  | 'i2i_editor'
  | 'iteration_check'
  | 'output'

export interface NodePosition {
  x: number
  y: number
}

export interface VLMDetectorData {
  cultural_score?: number
  prompt_score?: number
  issues: string[]
  suggestions: string[]
  text_kb_results: any[]
  clip_rag_results: any[]
  analysis_time?: number
}

export interface ReferenceSelectorData {
  candidates: any[]
  selected_reference?: any
  clip_scores: number[]
  keyword_matches: number[]
  final_scores: number[]
}

export interface I2IEditorData {
  model_name: string
  iteration: number
  current_step?: number
  total_steps?: number
  prompt_used?: string
  reference_image?: string
  editing_time?: number
  progress?: number
}

export interface T2IGeneratorData {
  model_name: string
  prompt: string
  generated_image?: string
  generation_time?: number
  parameters: Record<string, any>
}

export interface BaseNodeData {
  id: string
  type: NodeType
  label: string
  status: NodeStatus
  position: NodePosition
  data: Record<string, any>
  error?: string
  start_time?: number
  end_time?: number
  progress: number
}

export interface GPUStats {
  available: boolean
  device_name?: string
  current_device?: number
  gpu_count?: number
  memory?: {
    allocated_gb: number
    reserved_gb: number
    total_gb: number
    used_percent: number
  }
  utilization_percent?: number
  temperature_c?: number
  error?: string
  message?: string
}

export interface JobCreationMessage {
  status: 'no_gaps' | 'creating' | 'created' | 'skipped' | 'error'
  job_id?: string
  gap?: {
    category: string
    subcategory?: string
    keywords?: string[]
    priority?: number
  }
  message: string
  error?: string
}

export interface JobProposal {
  id: string  // Unique ID for this proposal
  category: string
  subcategory?: string
  keywords: string[]
  priority: number
  reason: string
}

export interface JobProposalMessage {
  proposals: JobProposal[]
  pipeline_id: string
}

export interface WebSocketMessage {
  type: 'node_update' | 'pipeline_status' | 'error' | 'gpu_stats' | 'job_creation' | 'job_proposal'
  node_id?: string
  status?: NodeStatus | PipelineStatus | string
  data?: Record<string, any>
  stats?: GPUStats
  gap?: JobCreationMessage['gap']
  job_id?: string
  error?: string
  message?: string
  timestamp: number
  current_node?: string
  progress?: number
  // For job_proposal messages
  proposals?: JobProposal[]
  pipeline_id?: string
}

export interface PipelineConfig {
  prompt: string
  country: string
  category: string
  t2i_model: string
  i2i_model: string
  max_iterations: number
  target_score: number
  load_in_4bit: boolean
}

export type PipelineStatus = 'idle' | 'running' | 'completed' | 'error'

export interface PipelineState {
  status: PipelineStatus
  current_node_id?: string
  current_iteration: number
  progress: number
  config?: PipelineConfig
  start_time?: number
  end_time?: number
  error?: string
  initial_image?: string
  final_image?: string
  iteration_images: string[]
  scores_history: Array<{ cultural: number; prompt: number }>
}

export interface HistorySummary {
  pipeline_id: string
  timestamp: number
  prompt: string
  country: string
  status: string
  duration?: number
  iterations: number
  final_score?: number
  t2i_model: string
  i2i_model: string
}

export interface PromptFlowStep {
  step: string
  prompt: string
  metadata: Record<string, any>
}

export interface RAGResult {
  query: string
  results: Array<Record<string, any>>
  top_k: number
  search_time: number
}

export interface NodeHistory {
  node_id: string
  node_type: string
  status: string
  start_time: number
  end_time?: number
  prompt_flow: PromptFlowStep[]
  text_rag?: RAGResult
  clip_rag?: RAGResult
  reference_candidates: Array<Record<string, any>>
  reference_scores: Record<string, any>
  selected_reference?: Record<string, any>
  vlm_analysis?: Record<string, any>
  editing_params?: Record<string, any>
  iteration?: number
  editing_prompt?: string
  issues_fixed?: number
  output_data: Record<string, any>
  error?: string
}

export interface PipelineHistory {
  pipeline_id: string
  status: string
  config: Record<string, any>
  start_time: number
  end_time?: number
  duration?: number
  initial_image_path?: string
  final_image_path?: string
  iteration_count: number
  final_cultural_score?: number
  final_prompt_score?: number
  nodes: Record<string, NodeHistory>
  scores_history: Array<Record<string, number>>
  jobs_created: Array<Record<string, any>>
}
