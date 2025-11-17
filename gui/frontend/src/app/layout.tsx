import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'CCUB2 Agent Pipeline',
  description: 'Node-based workflow visualization for CCUB2 cultural bias mitigation',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-background-primary text-text-primary">
        {children}
      </body>
    </html>
  )
}
