interface ModelService {
  initialize: () => Promise<void>;
  predict: (context: string) => Promise<string[]>;
  cleanup: () => Promise<void>;
}

class TinyLlamaService implements ModelService {
  private isInitialized: boolean = false;
  private apiUrl: string = 'http://localhost:3000/api';

  async initialize(): Promise<void> {
    try {
      const response = await fetch(`${this.apiUrl}/initialize`, { method: 'POST' });
      if (!response.ok) throw new Error('Failed to initialize model');
      this.isInitialized = true;
    } catch (error) {
      console.error('Model initialization failed:', error);
      throw error;
    }
  }

  async predict(context: string): Promise<string[]> {
    if (!this.isInitialized) {
      throw new Error('Model not initialized');
    }

    try {
      const response = await fetch(`${this.apiUrl}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ context })
      });

      if (!response.ok) throw new Error('Prediction failed');
      const predictions = await response.json();
      return predictions.words;
    } catch (error) {
      console.error('Prediction failed:', error);
      throw error;
    }
  }

  async cleanup(): Promise<void> {
    if (!this.isInitialized) return;

    try {
      const response = await fetch(`${this.apiUrl}/cleanup`, { method: 'POST' });
      if (!response.ok) throw new Error('Cleanup failed');
      this.isInitialized = false;
    } catch (error) {
      console.error('Model cleanup failed:', error);
      throw error;
    }
  }
}

export const modelService = new TinyLlamaService();