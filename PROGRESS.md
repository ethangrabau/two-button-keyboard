# Two-Button Keyboard BCI Project Progress

## Latest Updates (December 13, 2024)

### Project Evolution
We've explored several approaches:
1. Pattern matching with LLM candidate selection (archived)
   - Too slow (~2s latency)
   - Complex pipeline
   - Difficult to optimize

2. Current Approach: Direct Pattern Transformer
   - Direct pattern-to-token generation
   - Built-in L/R pattern constraints
   - No LLM dependency
   - Target latency: <300ms

### Current State
- Successfully implemented keyboard L/R pattern mapping (passing all tests)
- Built prototype transformer model with pattern constraints
- Implemented data loading pipeline for chat data
- Built comprehensive training infrastructure
- Architecture components:
  - Pattern Encoder: Converts L/R patterns to embeddings
  - Context Encoder: Handles chat history
  - Pattern-Constrained Decoder: Generates valid tokens

### Recent Additions
1. Data Processing:
   - Support for Google Messages JSON format
   - Support for preprocessed CSV format
   - Word frequency weighting
   - Efficient batch processing

2. Training Infrastructure:
   - Mixed precision training
   - Gradient accumulation
   - Learning rate warmup
   - Automated checkpointing
   - Privacy-focused (all local)

### Next Steps
1. Test Training Pipeline:
   - Run initial training loop
   - Validate data processing
   - Test checkpointing
   - Measure basic metrics

2. Model Optimization:
   - Ensure <300ms latency
   - Target 90%+ accuracy after user-specific training
   - Optimize for Metal acceleration

3. Create Evaluation Tools:
   - Pattern prediction accuracy
   - Latency benchmarking
   - Example generations

### Technical Details
- Location: src/direct_pattern_transformer/
- Python 3.10 with PyTorch
- Core Components:
  - model.py: Pattern-constrained transformer
  - keyboard_mapping.py: L/R pattern generation
  - data_loader.py: Chat data processing
  - trainer.py: Training infrastructure
  - test_model.py: Test suite

### Performance Targets
- Pattern Matching: <50ms
- Total Prediction: <300ms
- Accuracy: >90% (with user-specific training)
- Novel Word Support: Yes (through subword tokens)