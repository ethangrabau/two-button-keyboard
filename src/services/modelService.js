const API_URL = 'http://localhost:3000/api';

class ModelService {
  constructor() {
    this.isInitialized = false;
  }

  async initialize() {
    try {
      const response = await fetch(`${API_URL}/status`);
      const data = await response.json();
      this.isInitialized = data.initialized;
      return this.isInitialized;
    } catch (error) {
      console.error('Model initialization failed:', error);
      this.isInitialized = false;
      return false;
    }
  }

  async getPredictions(currentText, pattern) {
    if (!this.isInitialized) {
      console.warn('Model not initialized, attempting to initialize...');
      await this.initialize();
    }

    try {
      console.log('Requesting prediction for:', currentText);
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: currentText || '' }),
      });

      if (!response.ok) {
        throw new Error(`Prediction request failed: ${response.statusText}`);
      }

      const { prediction } = await response.json();
      console.log('Received prediction:', prediction);
      
      // Only do pattern filtering on the frontend
      if (pattern && prediction) {
        const keyboard = {
          left: ['Q', 'W', 'E', 'R', 'T', 'A', 'S', 'D', 'F', 'G', 'Z', 'X', 'C', 'V'],
          right: ['Y', 'U', 'I', 'O', 'P', 'H', 'J', 'K', 'L', 'B', 'N', 'M']
        };
        
        const predictionPattern = prediction
          .toUpperCase()
          .split('')
          .map(char => keyboard.left.includes(char) ? 'L' : 'R')
          .join('');
          
        if (predictionPattern.startsWith(pattern)) {
          return { predictions: [prediction] };
        }
        return { predictions: [] };
      }
      
      return { predictions: prediction ? [prediction] : [] };
    } catch (error) {
      console.error('Prediction error:', error);
      return { predictions: [] };
    }
  }

  cleanup() {
    this.isInitialized = false;
  }
}

export const modelService = new ModelService();