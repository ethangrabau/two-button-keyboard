import React, { useState } from 'react';

const KeyboardInterface = () => {
  const [pattern, setPattern] = useState('');
  
  // Define keyboard layouts by rows
  const keyboard = {
    left: {
      top: ['Q', 'W', 'E', 'R', 'T'],
      middle: ['A', 'S', 'D', 'F', 'G'],
      bottom: ['Z', 'X', 'C', 'V']
    },
    right: {
      top: ['Y', 'U', 'I', 'O', 'P'],
      middle: ['H', 'J', 'K', 'L'],
      bottom: ['B', 'N', 'M']
    }
  };
  
  const handleClick = (side) => {
    setPattern(prev => prev + (side === 'left' ? 'L' : 'R'));
  };

  const handleSpace = () => {
    setPattern(prev => prev + '_');
  };
  
  // Helper function to render a row of keys
  const renderKeyRow = (keys, extraClasses = '') => (
    <div className={`flex gap-2 justify-start ${extraClasses}`}>
      {keys.map(letter => (
        <div key={letter} className="w-10 h-10 bg-white rounded shadow flex items-center justify-center font-mono">
          {letter}
        </div>
      ))}
    </div>
  );

  return (
    <div className="p-4 max-w-3xl mx-auto">
      {/* Clear Pattern Button - Top Right */}
      <div className="flex justify-between mb-4">
        <div className="text-lg font-mono bg-gray-100 rounded-lg p-4 flex-grow mr-4">
          {pattern || 'No input yet'}
        </div>
        <button 
          onClick={() => setPattern('')}
          className="px-4 py-2 bg-red-100 rounded-lg hover:bg-red-200 transition-colors whitespace-nowrap"
        >
          Clear Pattern
        </button>
      </div>
      
      {/* Keyboard Layout */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        {/* Left Side */}
        <div 
          onClick={() => handleClick('left')}
          className="p-4 bg-blue-100 rounded-lg cursor-pointer hover:bg-blue-200 transition-colors"
        >
          <div className="flex flex-col gap-4">
            {renderKeyRow(keyboard.left.top, 'ml-0')}
            {renderKeyRow(keyboard.left.middle, 'ml-4')}
            {renderKeyRow(keyboard.left.bottom, 'ml-6')}
          </div>
        </div>
        
        {/* Right Side */}
        <div 
          onClick={() => handleClick('right')}
          className="p-4 bg-green-100 rounded-lg cursor-pointer hover:bg-green-200 transition-colors"
        >
          <div className="flex flex-col gap-4">
            {renderKeyRow(keyboard.right.top)}
            {renderKeyRow(keyboard.right.middle, 'ml-2')}
            {renderKeyRow(keyboard.right.bottom, 'ml-4')}
          </div>
        </div>
      </div>
      
      {/* Space Bar */}
      <button 
        onClick={handleSpace}
        className="w-full p-4 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors font-mono text-lg"
      >
        Space Bar
      </button>
    </div>
  );
};

export default KeyboardInterface;