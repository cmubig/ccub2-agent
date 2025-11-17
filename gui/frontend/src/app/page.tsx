'use client'

import React from 'react'
import PipelineCanvas from '@/components/Pipeline/PipelineCanvas'

export default function Home() {
  return (
    <main className="w-screen h-screen bg-background-primary">
      <PipelineCanvas />
    </main>
  )
}
