/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Dark Monotone Color Palette (NO GRADIENTS!)
        background: {
          primary: '#0a0a0a',    // Pure black background
          secondary: '#141414',  // Slightly lighter black
          tertiary: '#1e1e1e',   // Card/panel background
        },
        surface: {
          dark: '#0f0f0f',       // Dark surface
          base: '#1a1a1a',       // Base surface
          elevated: '#252525',   // Elevated surface
          hover: '#2a2a2a',      // Hover state
        },
        border: {
          dark: '#2a2a2a',       // Dark borders
          base: '#333333',       // Base borders
          light: '#404040',      // Light borders
          accent: '#ffffff',     // Accent borders (white)
        },
        text: {
          primary: '#ffffff',    // Primary text (white)
          secondary: '#a3a3a3',  // Secondary text (gray)
          tertiary: '#737373',   // Tertiary text (darker gray)
          disabled: '#525252',   // Disabled text
        },
        status: {
          pending: '#6b7280',    // Gray for pending
          processing: '#ffffff', // White for processing
          completed: '#ffffff',  // White for completed
          error: '#ef4444',      // Red for error (only colored element)
        },
        node: {
          bg: '#1a1a1a',         // Node background
          border: '#333333',     // Node border
          hover: '#252525',      // Node hover
          active: '#2a2a2a',     // Node active
        }
      },
      boxShadow: {
        'node': '0 0 0 1px rgba(255, 255, 255, 0.1)',
        'node-active': '0 0 0 2px rgba(255, 255, 255, 0.3)',
        'panel': '0 0 20px rgba(0, 0, 0, 0.5)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}
