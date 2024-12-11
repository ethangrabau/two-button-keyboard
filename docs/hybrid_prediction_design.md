# Hybrid Prediction System Design

## Overview
The Two-Button Keyboard uses a novel hybrid approach combining pattern matching with LLM-based selection for accurate and context-aware text prediction.

## Core Components

### 1. Pattern Matching (Phase 1 - Complete)
- Converts keyboard input into L/R patterns (e.g., "hello" -> "LRLLR")
- Maintains frequency-based word rankings
- Handles both single words and basic phrases
- Implemented in `PatternMatcher` and basic `PhraseMatcher` classes

### 2. LLM Integration (Phase 2 - In Progress)
- Local Phi-2/TinyLlama integration for intelligent word selection
- Basic infrastructure in place through `LLMManager`
- Runs locally on CPU/Metal acceleration

### 3. Hybrid Prediction System (Phase 2 - Design)

#### Pattern-to-Candidates Flow
1. User inputs L/R sequence (e.g., "RR RRL LLL RRR")
2. Pattern matcher generates candidates for each position:
   ```python
   {
     'position': 1,
     'pattern': 'RR',
     'candidates': ['hi', 'on', 'in', ...],
     'scores': [0.9, 0.7, 0.5, ...]
   }
   ```
3. LLM receives:
   - Complete candidate lists for each position
   - Message history context
   - Current input pattern
4. LLM selects most natural phrase
5. System validates selections against original patterns

#### Key Features
- Preserves all L/R signal information
- Uses LLM for semantic understanding
- Maintains fast local fallback
- Supports both BCI and normal keyboard modes

## Future Enhancements
- Learning system for new words/phrases
- Personal vocabulary integration
- Adaptive pattern recognition
- Context-based prediction improvements

## Implementation Status
- Phase 1: Complete
  - Pattern matching
  - Basic phrase handling
  - Frequency-based ranking

- Phase 2: In Progress
  - LLM infrastructure setup
  - Enhancing PhraseMatcher for candidate generation
  - Implementing hybrid selection system