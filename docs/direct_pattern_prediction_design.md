# Direct Pattern Prediction Design

## Overview
After testing various approaches including pattern matching, large language models (Phi-2, TinyLlama), and BERT-based candidate selection, we propose a new direct pattern prediction architecture focusing on sub-200ms latency and high accuracy by learning directly from user chat patterns.

## Core Innovation
Instead of generating word candidates and scoring them, the new approach directly predicts words/phrases from L/R patterns by learning the relationship between:
- Chat context (previous messages)
- L/R keyboard patterns
- Actual words/phrases used

## Architecture

### 1. Model Components

```python
class TwoButtonPredictor(nn.Module):
    def __init__(self):
        self.context_encoder = BertModel(...)  # Encode chat context
        self.pattern_encoder = nn.LSTM(...)    # Encode L/R patterns
        self.decoder = nn.TransformerDecoder(...) # Generate matching words
```

#### Context Encoder
- Pre-trained BERT model fine-tuned on chat data
- Encodes previous N messages into context embedding
- Caches results for performance

#### Pattern Encoder
- LSTM encoding of L/R sequences
- Each pattern position is represented as [L=0, R=1]
- Padded to fixed length for batching
- Learns common L/R patterns in user's typing

#### Word Decoder
- Transformer decoder architecture
- Takes context + pattern embeddings
- Outputs probability distribution over vocabulary
- Only predicts words matching input pattern

### 2. Training Pipeline

#### Data Preparation
- Extract from chat history:
  - Previous N messages (context)
  - Target message
  - L/R pattern for target
- Augment data with pattern variations
- Build user-specific vocabulary

#### Training Process
1. Encode chat context using BERT
2. Encode L/R pattern using LSTM
3. Generate word predictions
4. Compare with actual words used
5. Update weights to maximize accuracy

#### Loss Functions
- Cross entropy on word predictions
- Pattern matching regularization
- Length prediction auxiliary task

### 3. Performance Optimizations

#### Model Optimization
- Int8 quantization
- Metal acceleration on Mac
- Pruned architecture
- Distillation from larger model

#### Runtime Optimization
- Context caching
- Batched pattern processing
- Limited context window
- Vocabulary pruning
- Pattern matching pre-filtering

#### Continuous Learning
- Update on user corrections
- Track prediction success rate
- Adapt to changing patterns
- Profile-based models

## Performance Targets

### Latency Budget
1. Context Processing: <50ms
   - BERT encoding
   - Cache lookup
   - Context window management

2. Pattern Processing: <50ms
   - LSTM encoding
   - Pattern validation
   - Batch preparation

3. Word Generation: <100ms
   - Transformer decoding
   - Pattern matching
   - Vocabulary filtering

### Accuracy Targets
- Top-1 Accuracy: >80%
- Top-3 Accuracy: >95%
- Pattern Match Rate: 100%
- User Acceptance Rate: >90%

## Implementation Phases

### Phase 1: Core Model (Current)
1. Basic architecture implementation
2. Training pipeline setup
3. Initial model training
4. Latency benchmarking

### Phase 2: Optimization
1. Model quantization
2. Caching implementation
3. Metal acceleration
4. Performance tuning

### Phase 3: Continuous Learning
1. User feedback integration
2. Profile management
3. Adaptive training
4. Pattern optimization

## Technical Requirements

### Development Environment
- Python 3.10
- PyTorch with Metal support
- Transformers library
- Local filesystem for testing
- Express + Flask servers

### Production Environment
- Quantized models
- Optimized inference
- Profile management
- Monitoring system

## Next Steps
1. Prototype core model architecture
2. Implement training pipeline
3. Test with sample chat data
4. Measure baseline performance
5. Begin optimization process