import React, { useState, useRef, useEffect } from 'react';
import WordPrediction from './WordPrediction';

const KeyboardInterface = () => {
  const [pattern, setPattern] = useState('');
  const [currentText, setCurrentText] = useState('');
  const [messageHistory, setMessageHistory] = useState([]);
  
  // Queue state for batching pattern updates
  const queueTimeoutRef = useRef(null);
  const [pendingPattern, setPendingPattern] = useState('');
  
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

  // Process pending pattern updates
  useEffect(() => {
    if (pendingPattern !== pattern) {
      if (queueTimeoutRef.current) {
        clearTimeout(queueTimeoutRef.current);
      }
      
      queueTimeoutRef.current = setTimeout(() => {
        setPattern(pendingPattern);
      }, 50); // Short delay to batch updates
    }
    
    return () => {
      if (queueTimeoutRef.current) {
        clearTimeout(queueTimeoutRef.current);
      }
    };
  }, [pendingPattern]);
  
  const handleClick = (side) => {
    const newChar = side === 'left' ? 'L' : 'R';
    setPendingPattern(prev => prev + newChar);
  };

  const handleSpace = () => {
    if (pendingPattern || pattern) {
      setPendingPattern(prev => prev + ' ');
    }
  };

  const handleWordSelect = (word) => {
    setCurrentText(word);
    setPattern('');
    setPendingPattern('');
  };

  const handleSendMessage = () => {
    const messageToSend = currentText.trim();
    if (messageToSend) {
      setMessageHistory(prev => [...prev, messageToSend]);
      setCurrentText('');
      setPattern('');
      setPendingPattern('');
    }
  };

  const handleClear = () => {
    setPattern('');
    setPendingPattern('');
    setCurrentText('');
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

  // Display either pending or current pattern
  const displayPattern = pendingPattern || pattern;
  const isPhrasePrediction = displayPattern.includes(' ');

  return (
    <div className="p-4 max-w-3xl mx-auto">
      {/* Message History */}
      <div className="mb-4 p-4 bg-gray-50 rounded-lg max-h-48 overflow-y-auto">
        {messageHistory.map((message, index) => (
          <div key={index} className="mb-2 last:mb-0 p-2 bg-white rounded">
            {message}
          </div>
        ))}
        {messageHistory.length === 0 && (
          <div className="text-gray-400 italic">No messages yet</div>
        )}
      </div>

      {/* Current Text and Pattern Display */}
      <div className="flex justify-between mb-4">
        <div className="text-lg font-mono bg-gray-100 rounded-lg p-4 flex-grow mr-4 min-h-[4rem] flex items-center">
          {currentText}
          {displayPattern && (
            <span className={`ml-1 ${isPhrasePrediction ? 'text-green-500' : 'text-blue-500'} font-bold`}>
              {displayPattern}
            </span>
          )}
        </div>
        <button 
          onClick={handleClear}
          className="px-4 py-2 bg-red-100 rounded-lg hover:bg-red-200 transition-colors whitespace-nowrap"
        >
          Clear
        </button>
      </div>

      {/* Word Prediction */}
      <WordPrediction 
        pattern={pattern}
        currentText={currentText}
        onWordSelect={handleWordSelect}
        messageHistory={messageHistory}
        maxResults={isPhrasePrediction ? 3 : 5}
      />
      
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
      
      {/* Bottom Controls */}
      <div className="grid grid-cols-2 gap-4">
        <button 
          onClick={handleSpace}
          className={`p-4 rounded-lg transition-colors font-mono text-lg
            ${displayPattern ? 'bg-green-100 hover:bg-green-200' : 'bg-gray-100 hover:bg-gray-200'}`}
        >
          Add Space
        </button>
        <button 
          onClick={handleSendMessage}
          className="p-4 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-mono text-lg"
        >
          Send Message
        </button>
      </div>

      {/* Mode Indicator */}
      <div className="mt-2 text-center text-sm text-gray-500">
        {isPhrasePrediction ? 'Phrase Prediction' : 'Word Prediction'}
      </div>
    </div>
  );
};

export default KeyboardInterface;