import React, { useState, useEffect } from 'react';
import { predictionService } from '../services/predictionService';
import { benchmark } from '../utils/benchmark';

const WordPrediction = ({ pattern, currentText = '', onWordSelect, messageHistory = [] }) => {
  const [predictions, setPredictions] = useState([]);
  const [isModelReady, setIsModelReady] = useState(false);
  const [modelStatus, setModelStatus] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);
  const [isPhrase, setIsPhrase] = useState(false);

  // Initialize prediction service
  useEffect(() => {
    const initService = async () => {
      try {
        const success = await predictionService.initialize();
        setIsModelReady(success);
        setModelStatus(predictionService.getModelStatus());
        setError(null);
      } catch (err) {
        setError('Failed to initialize prediction service');
        console.error(err);
      }
    };
    initService();
    
    // Cleanup
    return () => {
      predictionService.cleanup();
    };
  }, []);

  // Get predictions when pattern changes
  useEffect(() => {
    if (!pattern || !isModelReady) return;

    const getPredictions = async () => {
      const startTime = performance.now();
      
      try {
        // Get the last few words as context
        const context = currentText.trim().split(/\s+/).slice(-3);
        
        const result = await predictionService.getPredictions(
          pattern,
          5,
          context,
          messageHistory
        );
        
        const endTime = performance.now();
        const latency = endTime - startTime;
        
        benchmark.recordPrediction({ latency, success: true });
        
        if (result.predictions?.length > 0) {
          setPredictions(result.predictions);
          setIsPhrase(result.isPhrase);
          setError(null);
        } else {
          setPredictions([]);
          setError('No predictions found');
        }
        
        // Update metrics
        setMetrics({
          ...benchmark.getStats(),
          patternStats: result.stats
        });
      } catch (err) {
        setError('Prediction failed');
        console.error(err);
        benchmark.recordPrediction({ latency: 0, success: false });
      }
    };

    getPredictions();
  }, [pattern, currentText, messageHistory, isModelReady]);

  // Render service status when no pattern
  if (!pattern) {
    return (
      <div className="mb-4 p-4 bg-gray-100 rounded-lg">
        <div className="flex justify-between items-center mb-2">
          <span className="text-gray-600">
            {isModelReady ? 
              "Type using left (L) and right (R) buttons" :
              "Initializing prediction service..."}
          </span>
          {modelStatus && (
            <div className="text-xs text-gray-500">
              <div>Device: {modelStatus.device}</div>
              <div>LLM: {modelStatus.status}</div>
              <div>Words: {modelStatus.wordsLoaded}</div>
            </div>
          )}
        </div>
        {error && (
          <div className="text-red-500 text-sm mt-2">{error}</div>
        )}
      </div>
    );
  }

  return (
    <div className="mb-4">
      <div className="text-sm text-gray-500 mb-2 flex justify-between items-center">
        <span>
          {isPhrase ? "Phrase Prediction" : "Word Prediction"} ({predictions.length})
        </span>
        {metrics && (
          <div className="flex gap-4 text-xs">
            <span>
              Latency: {metrics.averageLatency?.toFixed(1) || 0}ms
            </span>
            {metrics.patternStats && (
              <span>
                Words: {metrics.patternStats.total_words}
              </span>
            )}
          </div>
        )}
      </div>
      
      {error ? (
        <div className="text-red-500 text-sm mb-2">{error}</div>
      ) : null}
      
      <div className="flex flex-wrap gap-2">
        {predictions.map((word, index) => (
          <button
            key={word + index}
            onClick={() => onWordSelect(word)}
            className={`px-3 py-1 rounded-full shadow-sm hover:shadow-md 
                     transition-shadow border text-sm
                     ${isPhrase ? 
                       'bg-blue-50 border-blue-200 hover:bg-blue-100' :
                       'bg-white border-gray-200 hover:bg-gray-50'}`}
          >
            {word}
          </button>
        ))}
        {predictions.length === 0 && !error && (
          <div className="text-gray-400 italic">No matches found</div>
        )}
      </div>
    </div>
  );
};

export default WordPrediction;