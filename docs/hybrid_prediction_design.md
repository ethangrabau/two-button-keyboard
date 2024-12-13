# Hybrid Prediction System Design

## Overview
The Two-Button Keyboard uses a novel hybrid approach combining pattern matching with transformer-based sequence prediction for accurate and context-aware text generation.

## Core Components

### 1. Pattern Matching (Phase 1 - Complete)
- Converts keyboard input into L/R patterns (e.g., "hello" -> "LRLLR")
- Maintains frequency-based word rankings
- Handles both single words and basic phrases
- Implemented in `PatternMatcher` and basic `PhraseMatcher` classes

### 2. Neural Prediction System (Phase 2 - New Design)

#### Architecture
- BERT-based sequence prediction model
- Optimized for sub-100ms inference
- Supports continuous learning from user interactions
- Focuses on selecting from pre-filtered word candidates

#### Model Design
Input Format:
```
[CLS] context_msg1 [SEP] context_msg2 [SEP] word1_candidates word2_candidates [MASK] [MASK]
```

Example:
```
[CLS] how are you [SEP] i am good [SEP] hi|hello|hey what|why|when [MASK] [MASK]
```

Model Tasks:
1. Primary: Select most natural word sequence from candidates
2. Secondary: Learn user's writing style and preferences
3. Optional: Predict likely next words for faster future input

Performance Targets:
- Pattern Matching: <30ms (keyboard feedback)
- BERT Inference: 50-100ms
- Total Prediction: <200ms
- Memory Usage: <500MB

### 3. Continuous Learning System (Phase 3 - Planned)

#### Training Pipeline
1. Initial Model:
   - Pre-trained BERT base
   - Fine-tuned on conversation data
   - Optimized for word sequence selection

2. User Adaptation:
   - Log successful predictions
   - Store user message history
   - Periodic retraining (e.g., nightly)
   - Profile-based models (work/casual/etc)

3. Performance Optimization:
   - INT8 quantization
   - Kernel fusion for Metal
   - Aggressive caching
   - Prediction prefetching

## Implementation Phases

### Phase 1: Complete
- Pattern matching
- Basic phrase handling
- Frequency-based ranking

### Phase 2: In Progress
- BERT model integration
- Fast candidate selection
- Context handling
- Basic performance optimization

### Phase 3: Planned
- Continuous learning pipeline
- User profiles
- Advanced caching
- Mobile optimization

## Performance Considerations

### Latency Budget
1. Immediate Feedback (<30ms):
   - Button press registration
   - Pattern matching update
   - Candidate generation

2. Fast Prediction (50-100ms):
   - BERT inference
   - Context processing
   - Initial word selection

3. Background Tasks (>100ms):
   - Training data collection
   - Cache updates
   - Model fine-tuning

### Memory Management
- Model quantization (INT8)
- Efficient context window
- Smart cache eviction
- Profile swapping