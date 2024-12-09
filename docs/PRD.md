# Two-Button Keyboard - Product Requirements Document

## Overview
An experimental keyboard interface that simplifies text input to just two buttons, using advanced prediction to determine intended words based on which side of a QWERTY keyboard the letters appear on.

## Core Features

### 1. User Interface
- Two main buttons representing left/right sides of QWERTY keyboard
- Visual keyboard layout showing the split
- Word prediction display area
- "Accept" button for confirming predictions
- Clear visual feedback for button presses
- Display of current input pattern and predictions

### 2. Input Mechanism
- Left button maps to: Q,W,E,R,T,A,S,D,F,G,Z,X,C,V,B
- Right button maps to: Y,U,I,O,P,H,J,K,L,N,M
- Pattern recording (e.g., "HELLO" = L-L-L-L-R)
- Support for basic punctuation and space
- Clear/backspace functionality

### 3. Prediction System
- Base prediction on common English words initially
- Support for fine-tuning with user's message history
- Real-time prediction updates as pattern is entered
- Ranking system for multiple possible matches
- Frequency-based word prioritization

## Technical Requirements

### Phase 1: Basic Implementation
1. React web application setup
2. Basic UI components
3. Pattern recording system
4. Simple dictionary-based prediction
5. Basic word selection interface

### Phase 2: Enhanced Prediction
1. Integration with lightweight language model
2. Message history processing pipeline
3. Model fine-tuning system
4. Improved prediction ranking
5. Performance optimization

### Phase 3: Refinement
1. User testing and feedback
2. Performance metrics collection
3. UI/UX improvements
4. Mobile optimization
5. Potential custom model training

## Success Metrics
1. Input Speed (WPM)
2. Prediction Accuracy
3. Learning Curve Duration
4. User Satisfaction
5. Error Rate

## Project Timeline
- Phase 1: 1-2 weeks
- Phase 2: 2-3 weeks
- Phase 3: 1-2 weeks

## Technical Stack
- Frontend: React
- State Management: React Context/Redux
- Language Model: TBD (considering GPT-2 Tiny or custom n-gram)
- Styling: Tailwind CSS
- Build Tools: Vite