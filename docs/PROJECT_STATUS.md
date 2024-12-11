# Two Button Keyboard Project Status

## Current Status (December 11, 2024)

### Latest Decisions
- Modified architecture to use hybrid word/phrase prediction:
  1. Fast pattern matcher generates word candidates
  2. Phi-2 selects best phrases from candidates
  3. Context integration from message history
  4. Two-mode operation (normal keyboard/BCI)

### Architecture Implementation
- Pattern matching system complete:
  - L/R pattern generation
  - Word candidate generation
  - Fast lookup (<50ms)
  - Basic word prediction

- React frontend working:
  - Two-button interface
  - Pattern display
  - Word predictions
  - Message history

- Initial backend structure:
  - Express server with prediction endpoint
  - Flask server for ML components
  - Pattern matching API

### Completed
1. Basic Pattern Matcher:
   - L/R pattern generation
   - Word candidate lookup
   - Pattern verification
   - Test framework

2. Frontend Interface:
   - Two-button layout
   - Pattern visualization
   - Prediction display
   - Message history
   - Space/Clear functions

3. Backend Structure:
   - Express server setup
   - Flask ML server
   - Basic prediction API
   - CORS and error handling

4. Project Planning:
   - Detailed PRD created
   - Architecture defined
   - Implementation phases set
   - Test framework established

### Next Steps

#### Phase 2: Phrase Integration
1. Implement pattern sequence handling
2. Create Phi-2 selection system
3. Add message history tracking
4. Setup context integration

#### Phase 3: Learning System
1. Implement two-mode operation
2. Add vocabulary learning
3. Create phrase caching
4. Optimize performance

### Technical Details

#### Frontend Stack
- React + Vite
- Tailwind CSS
- TypeScript

#### Backend Stack
- Express (Node.js)
- Flask (Python)
- Pattern matcher
- Phi-2 (2.7B parameters)

#### Model Details
- Primary: Microsoft Phi-2
- Size: 2.7B parameters
- Metal acceleration on M3
- Target latency: 
  - Pattern matching: <50ms
  - Phrase prediction: <300ms

#### API Endpoints
- GET /api/status - System status
- POST /api/predict - Get predictions
  - Input: { pattern: string, context: string[] }
  - Output: { predictions: string[], confidence: number }

## Development Setup
1. Install dependencies:
```bash
npm install
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. Start Node backend: `npm run server`
3. Start ML server: `cd server && python prediction_server.py`
4. Start frontend: `npm run dev`