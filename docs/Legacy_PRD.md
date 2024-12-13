# Two-Button Keyboard Project PRD

## Overview
A novel text input system that uses only two buttons, designed as a stepping stone toward a brain-computer interface (BCI) for text input. The system uses a hybrid approach combining pattern matching, n-gram models, and advanced language models to predict words and phrases from minimal binary input.

## Vision
Create a text input system that can be controlled with minimal binary signals (initially two buttons, eventually brain signals) while maintaining high typing efficiency through intelligent phrase prediction. This serves as a prototype for future BCI text input systems.

## Target Users
1. Initial Phase: Users testing the two-button interface
2. Future Phase: Users of non-invasive BCI devices
3. Ultimate Goal: Users who need alternative text input methods due to mobility limitations

## Core Requirements

### Input System
- Two-button interface (Left/Right)
- Each button represents half of a QWERTY keyboard
- Pattern creation through sequences of L/R presses
- Visual feedback of current pattern
- Clear button for pattern reset
- Space functionality for word/phrase completion
- Seamless transition between manual and predictive modes

### Word/Phrase Prediction
- Multi-level prediction system:
  1. Fast pattern matching with candidate generation (<50ms)
  2. LLM-based phrase selection from candidates
  3. Context integration from message history
- Support for both single-word and multi-word predictions
- Predict based on:
  - Current L/R pattern sequence
  - Candidate words matching patterns
  - Previous messages (context)
  - User's typing history
- Fast inference time (varied by mode)
- High prediction accuracy with minimal input

### Learning & Adaptation
- Two-mode operation:
  - Normal keyboard mode for adding vocabulary
  - BCI mode using learned patterns
- Learn from user's normal typing
- Store personal vocabulary and phrases
- Cache frequent predictions
- Track context-specific patterns

### User Interface
- Clear display of current input pattern
- Word/phrase predictions
- Message history view
- Visual keyboard layout reference
- Loading states and error handling
- Responsive design for various devices

## Technical Requirements

### Model Requirements
- Hybrid prediction system (<2GB total size)
- Pattern matching: <50ms latency
- LLM inference: <300ms latency
- Support for efficient fine-tuning
- Local inference capability
- CPU/Metal acceleration support
- Potential for mobile/edge deployment

### System Architecture
- Frontend: React + Vite for UI
- Backend: Express server for model serving
- Multi-level prediction system:
  - Pattern matcher for candidate generation
  - Phi-2 for context-aware selection
  - Personal vocabulary integration
- WebSocket support for real-time updates
- Efficient data storage for personalization
- Cross-platform compatibility

### Performance Targets
- Pattern matching latency: <50ms
- LLM inference latency: <300ms
- Model load time: <5s
- Memory usage: <4GB
- Storage for personalization: <100MB
- Cache hit rate: >80% for common patterns

## Future Extensions

### BCI Integration
- Support for non-invasive BCI input
- Adaptable to various BCI signal types
- Configurable signal processing
- Robust error handling for noisy signals
- Calibration system for BCI input

### Advanced Features
- Full phrase prediction
- Topic detection and adaptation
- Emoji and special character support
- Multiple language support
- Custom vocabulary addition
- Prediction confidence indicators

## Success Metrics
1. Word/Phrase Prediction Accuracy
   - >80% accuracy for pattern matching
   - >60% accuracy for first phrase prediction
   - >90% accuracy within top 3 predictions

2. Input Efficiency
   - Average <3 button presses per word
   - >15 words per minute after practice
   - <5% error rate in pattern recognition
   - >80% cache hit rate for common patterns

3. User Experience
   - <1 hour learning curve
   - Pattern matching <50ms
   - Phrase prediction <300ms
   - >80% user satisfaction rating

## Implementation Phases

1. Phase 1: Basic Pattern Prediction
   - L/R pattern matching
   - Word candidate generation
   - Single word prediction
   - Fast response time

2. Phase 2: Phrase Integration
   - Pattern sequence handling
   - Phi-2 candidate selection
   - Message history tracking
   - Basic context understanding

3. Phase 3: Learning System
   - Two-mode operation
   - Personal vocabulary learning
   - Phrase caching
   - Performance optimization

## Future Phases
1. BCI signal processing integration
2. Advanced personalization features
3. Mobile/edge deployment
4. Multi-language support
5. Accessibility optimizations