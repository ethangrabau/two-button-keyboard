import React from 'react'
import KeyboardInterface from './components/KeyboardInterface'

function App() {
  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        <h1 className="text-3xl font-bold text-center mb-8">Two Button Keyboard</h1>
        <KeyboardInterface />
      </div>
    </div>
  )
}

export default App