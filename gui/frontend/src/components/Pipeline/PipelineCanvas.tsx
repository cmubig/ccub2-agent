'use client'

import React, { useCallback, useState, useEffect } from 'react'
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  MiniMap,
  Panel,
} from 'reactflow'
import 'reactflow/dist/style.css'

import { useWebSocket } from '@/hooks/useWebSocket'
import { pipelineAPI, historyAPI, jobsAPI } from '@/lib/api'
import { WebSocketMessage, NodeStatus, PipelineConfig, GPUStats, JobCreationMessage, PipelineHistory, JobProposal } from '@/lib/types'
import BaseNode from '@/components/Nodes/BaseNode'
import VLMDetectorNode from '@/components/Nodes/VLMDetectorNode'
import I2IEditorNode from '@/components/Nodes/I2IEditorNode'
import ConfigSidebar from '@/components/Panels/ConfigSidebar'
import NodeDetailPanel from '@/components/Panels/NodeDetailPanel'
import SystemStatusPanel from '@/components/Panels/SystemStatusPanel'
import HistoryPanel from '@/components/Panels/HistoryPanel'
import JobCreationToast from '@/components/Notifications/JobCreationToast'
import JobApprovalPanel from '@/components/Panels/JobApprovalPanel'

// Define custom node types
const nodeTypes = {
  base: BaseNode,
  vlm_detector: VLMDetectorNode,
  i2i_editor: I2IEditorNode,
}

// Initial nodes configuration
const initialNodes: Node[] = [
  {
    id: 'input',
    type: 'base',
    position: { x: 450, y: 0 },
    data: {
      label: 'Input',
      status: 'pending' as NodeStatus,
      nodeType: 'input',
    },
  },
  {
    id: 't2i_generator',
    type: 'base',
    position: { x: 350, y: 100 },
    data: {
      label: 'T2I Generator',
      status: 'pending' as NodeStatus,
      nodeType: 't2i_generator',
    },
  },
  {
    id: 'vlm_detector',
    type: 'vlm_detector',
    position: { x: 400, y: 480 },
    data: {
      label: 'VLM Detector',
      status: 'pending' as NodeStatus,
      nodeType: 'vlm_detector',
      culturalScore: null,
      promptScore: null,
      issues: [],
    },
  },
  {
    id: 'text_kb_query',
    type: 'base',
    position: { x: 100, y: 720 },
    data: {
      label: 'Text KB Query',
      status: 'pending' as NodeStatus,
      nodeType: 'text_kb_query',
    },
  },
  {
    id: 'clip_rag_search',
    type: 'base',
    position: { x: 700, y: 720 },
    data: {
      label: 'CLIP RAG Search',
      status: 'pending' as NodeStatus,
      nodeType: 'clip_rag_search',
    },
  },
  {
    id: 'reference_selector',
    type: 'base',
    position: { x: 400, y: 860 },
    data: {
      label: 'Reference Selector',
      status: 'pending' as NodeStatus,
      nodeType: 'reference_selector',
    },
  },
  {
    id: 'prompt_adapter',
    type: 'base',
    position: { x: 100, y: 1000 },
    data: {
      label: 'Prompt Adapter',
      status: 'pending' as NodeStatus,
      nodeType: 'prompt_adapter',
    },
  },
  {
    id: 'i2i_editor',
    type: 'i2i_editor',
    position: { x: 350, y: 1140 },
    data: {
      label: 'I2I Editor',
      status: 'pending' as NodeStatus,
      nodeType: 'i2i_editor',
      progress: 0,
      currentStep: 0,
      totalSteps: 28,
    },
  },
  {
    id: 'iteration_check',
    type: 'base',
    position: { x: 450, y: 1520 },
    data: {
      label: 'Iteration Check',
      status: 'pending' as NodeStatus,
      nodeType: 'iteration_check',
    },
  },
  {
    id: 'output',
    type: 'base',
    position: { x: 450, y: 1640 },
    data: {
      label: 'Output',
      status: 'pending' as NodeStatus,
      nodeType: 'output',
    },
  },
]

// Initial edges configuration
const initialEdges: Edge[] = [
  { id: 'e-input-t2i', source: 'input', target: 't2i_generator', animated: false },
  { id: 'e-t2i-vlm', source: 't2i_generator', target: 'vlm_detector', animated: false },
  { id: 'e-vlm-text', source: 'vlm_detector', target: 'text_kb_query', animated: false },
  { id: 'e-vlm-clip', source: 'vlm_detector', target: 'clip_rag_search', animated: false },
  { id: 'e-text-ref', source: 'text_kb_query', target: 'reference_selector', animated: false },
  { id: 'e-clip-ref', source: 'clip_rag_search', target: 'reference_selector', animated: false },
  { id: 'e-ref-adapter', source: 'reference_selector', target: 'prompt_adapter', animated: false },
  { id: 'e-ref-i2i', source: 'reference_selector', target: 'i2i_editor', animated: false },
  { id: 'e-adapter-i2i', source: 'prompt_adapter', target: 'i2i_editor', animated: false },
  { id: 'e-i2i-check', source: 'i2i_editor', target: 'iteration_check', animated: false },
  { id: 'e-check-output', source: 'iteration_check', target: 'output', animated: false },
  // Loop back edge
  { id: 'e-check-vlm', source: 'iteration_check', target: 'vlm_detector', animated: false, style: { stroke: '#6b7280', strokeDasharray: '5,5' } },
]

export default function PipelineCanvas() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)
  const [isConfigOpen, setIsConfigOpen] = useState(true)
  const [isPipelineRunning, setIsPipelineRunning] = useState(false)
  const [gpuStats, setGpuStats] = useState<GPUStats | null>(null)
  const [jobNotifications, setJobNotifications] = useState<JobCreationMessage[]>([])
  const [isHistoryVisible, setIsHistoryVisible] = useState(false)
  const [selectedHistory, setSelectedHistory] = useState<PipelineHistory | null>(null)
  const [jobProposals, setJobProposals] = useState<JobProposal[]>([])
  const [currentPipelineId, setCurrentPipelineId] = useState<string | null>(null)

  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback(
    (message: WebSocketMessage) => {
      console.log('WebSocket message:', message)

      if (message.type === 'node_update' && message.node_id) {
        // Update node data
        setNodes((nds) =>
          nds.map((node) => {
            if (node.id === message.node_id) {
              return {
                ...node,
                data: {
                  ...node.data,
                  status: message.status || node.data.status,
                  ...message.data,
                },
              }
            }
            return node
          })
        )

        // Animate edge for active node
        if (message.status === 'processing') {
          setEdges((eds) =>
            eds.map((edge) => ({
              ...edge,
              animated: edge.target === message.node_id,
            }))
          )
        }
      } else if (message.type === 'pipeline_status') {
        setIsPipelineRunning(message.status === 'running')
      } else if (message.type === 'gpu_stats' && message.stats) {
        setGpuStats(message.stats)
      } else if (message.type === 'job_creation') {
        // Add job creation notification
        const notification: JobCreationMessage = {
          status: message.status as JobCreationMessage['status'],
          job_id: message.job_id,
          gap: message.gap,
          message: message.message || 'Job creation update',
          error: message.error,
        }

        setJobNotifications((prev) => [...prev, notification])

        // Auto-dismiss after 10 seconds (except errors)
        if (message.status !== 'error') {
          setTimeout(() => {
            setJobNotifications((prev) =>
              prev.filter((n) => n !== notification)
            )
          }, 10000)
        }
      } else if (message.type === 'job_proposal') {
        // Handle job proposals
        if (message.proposals && message.pipeline_id) {
          setJobProposals(message.proposals)
          setCurrentPipelineId(message.pipeline_id)
        }
      }
    },
    [setNodes, setEdges]
  )

  const { isConnected, error: wsError } = useWebSocket(handleWebSocketMessage)

  // Handle edge connections
  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  )

  // Handle node click
  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    setSelectedNode(node)
  }, [])

  // Start pipeline
  const handleStartPipeline = async (config: PipelineConfig, imageFile: File | null) => {
    try {
      // Reset all nodes
      setNodes((nds) =>
        nds.map((node) => ({
          ...node,
          data: {
            ...node.data,
            status: 'pending' as NodeStatus,
          },
        }))
      )

      const formData = new FormData();
      formData.append('config', JSON.stringify(config));
      if (imageFile) {
        formData.append('image_file', imageFile);
      }

      const result = await pipelineAPI.start(formData)
      console.log('Pipeline started:', result)
      setIsPipelineRunning(true)
      setCurrentPipelineId(result.pipeline_id)
      // Reset job proposals when starting new pipeline
      setJobProposals([])
      // Keep config sidebar open so user can see settings
    } catch (error) {
      console.error('Failed to start pipeline:', error)
      alert('Failed to start pipeline. Check console for details.')
    }
  }

  // Stop pipeline
  const handleStopPipeline = async () => {
    try {
      await pipelineAPI.stop()
      setIsPipelineRunning(false)
    } catch (error) {
      console.error('Failed to stop pipeline:', error)
    }
  }

  // Dismiss job notification
  const handleDismissNotification = useCallback((index: number) => {
    setJobNotifications((prev) => prev.filter((_, i) => i !== index))
  }, [])

  // Handle job approval
  const handleApproveJobs = async (proposalIds: string[]) => {
    if (!currentPipelineId) return

    try {
      await jobsAPI.approve(proposalIds, currentPipelineId)
      // Close panel after successful approval
      setJobProposals([])
    } catch (error) {
      console.error('Failed to approve jobs:', error)
      alert('Failed to approve jobs. Check console for details.')
    }
  }

  // Handle job rejection
  const handleRejectJobs = async (proposalIds: string[]) => {
    if (!currentPipelineId) return

    try {
      await jobsAPI.reject(proposalIds, currentPipelineId)
      // Close panel after successful rejection
      setJobProposals([])
    } catch (error) {
      console.error('Failed to reject jobs:', error)
      alert('Failed to reject jobs. Check console for details.')
    }
  }

  // Handle history selection
  const handleSelectHistory = async (pipelineId: string) => {
    try {
      const history = await historyAPI.get(pipelineId)
      setSelectedHistory(history)
      console.log('Selected history:', history)

      // Load history data into nodes for visualization
      if (history.nodes) {
        setNodes((nds) =>
          nds.map((node) => {
            const nodeHistory = history.nodes[node.id]
            if (!nodeHistory) return node

            // Determine node status from history
            let status: NodeStatus = 'pending'
            if (nodeHistory.status === 'completed') {
              status = 'completed'
            } else if (nodeHistory.status === 'error') {
              status = 'error'
            } else if (nodeHistory.status === 'processing') {
              status = 'processing'
            }

            // Build updated node data
            const updatedData: any = {
              ...node.data,
              status,
            }

            // VLM Detector specific data
            if (node.id === 'vlm_detector' && nodeHistory.vlm_analysis) {
              updatedData.cultural_score = nodeHistory.vlm_analysis.cultural_score
              updatedData.prompt_score = nodeHistory.vlm_analysis.prompt_score
              updatedData.issues = nodeHistory.vlm_analysis.issues || []
              updatedData.iteration = nodeHistory.iteration
            }

            // I2I Editor specific data
            if (node.id === 'i2i_editor') {
              updatedData.iteration = nodeHistory.iteration
              if (nodeHistory.editing_prompt) {
                updatedData.editing_prompt = nodeHistory.editing_prompt
              }
            }

            // Prompt Adapter specific data
            if (node.id === 'prompt_adapter' && nodeHistory.prompt_flow) {
              const originalStep = nodeHistory.prompt_flow.find((s: any) => s.step === 'original')
              const adaptedStep = nodeHistory.prompt_flow.find((s: any) => s.step === 'adapted')

              if (originalStep) {
                updatedData.original_prompt = originalStep.prompt
              }
              if (adaptedStep) {
                updatedData.adapted_prompt = adaptedStep.prompt
                updatedData.model = adaptedStep.metadata?.model || history.config?.i2i_model
              }
              if (nodeHistory.issues_addressed !== undefined) {
                updatedData.issues_addressed = nodeHistory.issues_addressed
              }
            }

            // CLIP RAG Search specific data
            if (node.id === 'clip_rag_search' && nodeHistory.clip_rag) {
              updatedData.search_results = nodeHistory.clip_rag.results || nodeHistory.clip_rag
            }

            // Text KB Query specific data
            if (node.id === 'text_kb_query' && nodeHistory.text_rag) {
              updatedData.kb_results = nodeHistory.text_rag.results || nodeHistory.text_rag
            }

            // Reference Selector specific data
            if (node.id === 'reference_selector' && nodeHistory.selected_reference) {
              updatedData.selected = nodeHistory.selected_reference.selected || nodeHistory.selected_reference
              updatedData.score = nodeHistory.selected_reference.score
            }

            // Load images for T2I and I2I nodes
            if (node.id === 't2i_generator' || node.id === 'i2i_editor') {
              // Check output_data for image path (backend saves it there)
              const imagePath = nodeHistory.output_data?.image_path || nodeHistory.output_data?.output_image
              if (imagePath) {
                updatedData.has_image = true
                updatedData.image_path = imagePath
              }
            }

            return {
              ...node,
              data: updatedData,
            }
          })
        )
      }

    } catch (error) {
      console.error('Failed to load history:', error)
      alert('Failed to load history. Check console for details.')
    }
  }

  // Toggle history panel
  const handleToggleHistory = () => {
    setIsHistoryVisible(!isHistoryVisible)
  }

  return (
    <div className="w-full h-full relative">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        fitView
        className="bg-background-primary"
      >
        <Background color="#1a1a1a" gap={16} size={1} />
        <Controls className="bg-surface-elevated border border-border-base" />
        <MiniMap
          nodeColor={(node) => {
            switch (node.data.status) {
              case 'completed':
                return '#ffffff'
              case 'processing':
                return '#ffffff'
              case 'error':
                return '#ef4444'
              default:
                return '#6b7280'
            }
          }}
          className="bg-surface-elevated border border-border-base"
        />

        {/* Top Panel: Pipeline Status */}
        <Panel position="top-center" className="bg-surface-elevated border border-border-base rounded px-4 py-2">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-white' : 'bg-status-error'}`} />
              <span className="text-sm text-text-secondary">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>

            {isPipelineRunning && (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-white animate-pulse-slow" />
                <span className="text-sm text-white">Pipeline Running</span>
              </div>
            )}

            <button
              onClick={() => setIsConfigOpen(!isConfigOpen)}
              className="px-3 py-1 text-sm bg-neutral-800 border border-neutral-700 rounded hover:bg-neutral-700 transition-colors"
            >
              {isConfigOpen ? '⚙️ Hide Config' : '⚙️ Show Config'}
            </button>

            {isPipelineRunning && (
              <button
                onClick={handleStopPipeline}
                className="px-3 py-1 text-sm bg-status-error border border-status-error rounded hover:opacity-80 transition-opacity"
              >
                Stop Pipeline
              </button>
            )}
          </div>
        </Panel>
      </ReactFlow>

      {/* System Status Panel - Always visible when pipeline running */}
      {isPipelineRunning && (
        <div className="absolute top-4 left-4 w-72 z-10 slide-in">
          <SystemStatusPanel gpuStats={gpuStats} />
        </div>
      )}

      {/* Configuration Sidebar */}
      <ConfigSidebar
        onStart={handleStartPipeline}
        isVisible={isConfigOpen}
        onToggle={() => setIsConfigOpen(!isConfigOpen)}
      />

      {/* Node Detail Panel with Overlay */}
      {selectedNode && (
        <>
          {/* Dark overlay background */}
          <div
            className="fixed inset-0 bg-black/60 z-40 animate-fade-in"
            onClick={() => setSelectedNode(null)}
          />

          {/* Node Detail Panel */}
          <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] max-h-[90vh] z-50 animate-scale-in">
            <NodeDetailPanel node={selectedNode} onClose={() => setSelectedNode(null)} />
          </div>
        </>
      )}

      {/* Job Creation Notifications */}
      <JobCreationToast
        notifications={jobNotifications}
        onDismiss={handleDismissNotification}
      />

      {/* History Panel */}
      <HistoryPanel
        isVisible={isHistoryVisible}
        onToggle={handleToggleHistory}
        onSelectHistory={handleSelectHistory}
      />

      {/* Job Approval Panel */}
      {jobProposals.length > 0 && currentPipelineId && (
        <JobApprovalPanel
          proposals={jobProposals}
          pipelineId={currentPipelineId}
          onApprove={handleApproveJobs}
          onReject={handleRejectJobs}
          onClose={() => setJobProposals([])}
        />
      )}
    </div>
  )
}
