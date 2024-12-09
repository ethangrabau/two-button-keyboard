import React, { useState } from 'react';

const KeyboardInterface = () => {
  const [pattern, setPattern] = useState('');
  
  // Define keyboard layouts by rows - moved 'B' to right side
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