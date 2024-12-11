/**
 * Service for handling word predictions from the Python backend.
 */

const API_BASE_URL = 'http://localhost:5001/api';

class PredictionService {
    constructor() {
        this.isInitialized = false;
        this.modelStatus = null;
    }

    async initialize() {
        try {
            const response = await fetch(`${API_BASE_URL}/health`);
            const data = await response.json();
            
            this.isInitialized = data.status === 'healthy';
            this.modelStatus = {
                device: data.llm_device,
                status: data.llm_status,
                patternsLoaded: data.patterns_loaded,
                wordsLoaded: data.words_loaded
            };
            
            console.log('Prediction service initialized:', this.modelStatus);
            return this.isInitialized;
        } catch (error) {
            console.error('Failed to initialize prediction service:', error);
            return false;
        }
    }

    async getPredictions(pattern, maxResults = 5, context = [], messageHistory = []) {
        if (!this.isInitialized) {
            throw new Error('Prediction service not initialized');
        }

        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    pattern,
                    max_results: maxResults,
                    context: context,
                    message_history: messageHistory.slice(-3) // Last 3 messages for context
                })
            });

            const data = await response.json();
            if (!data.success) {
                throw new Error(data.error || 'Prediction failed');
            }

            return {
                predictions: data.predictions,
                stats: data.stats,
                isPhrase: data.is_phrase
            };
        } catch (error) {
            console.error('Prediction error:', error);
            return { predictions: [], stats: null, isPhrase: false };
        }
    }

    async cleanup() {
        if (!this.isInitialized) return;
        
        try {
            const response = await fetch(`${API_BASE_URL}/cleanup`, {
                method: 'POST'
            });
            const data = await response.json();
            
            if (!data.success) {
                console.error('Cleanup failed:', data.error);
            }
        } catch (error) {
            console.error('Cleanup error:', error);
        } finally {
            this.isInitialized = false;
            this.modelStatus = null;
        }
    }

    getModelStatus() {
        return this.modelStatus;
    }
}

export const predictionService = new PredictionService();