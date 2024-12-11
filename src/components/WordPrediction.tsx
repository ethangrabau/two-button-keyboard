import React, { useEffect, useState } from 'react';
import { modelService } from '../services/modelService';

interface WordPredictionProps {
  context: string;
  onPrediction?: (words: string[]) => void;
  onError?: (error: Error) => void;
}

export const WordPrediction: React.FC<WordPredictionProps> = ({
  context,
  onPrediction,
  onError
}) => {
  const [predictions, setPredictions] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const initializeModel = async () => {
      try {
        await modelService.initialize();
      } catch (err) {
        const error = err instanceof Error ? err : new Error('Failed to initialize model');
        setError(error);
        onError?.(error);
      }
    };

    initializeModel();

    return () => {
      modelService.cleanup().catch(err => {
        console.error('Cleanup failed:', err);
      });
    };
  }, []);

  useEffect(() => {
    const getPredictions = async () => {
      if (!context) return;

      setLoading(true);
      setError(null);

      try {
        const words = await modelService.predict(context);
        setPredictions(words);
        onPrediction?.(words);
      } catch (err) {
        const error = err instanceof Error ? err : new Error('Prediction failed');
        setError(error);
        onError?.(error);
      } finally {
        setLoading(false);
      }
    };

    getPredictions();
  }, [context, onPrediction, onError]);

  if (error) {
    return (
      <div className="text-red-600">
        Error: {error.message}
      </div>
    );
  }

  if (loading) {
    return <div className="text-gray-600">Loading predictions...</div>;
  }

  return (
    <div className="space-y-2">
      <h3 className="text-lg font-semibold">Predictions</h3>
      <ul className="space-y-1">
        {predictions.map((word, index) => (
          <li
            key={`${word}-${index}`}
            className="px-3 py-1 bg-blue-50 rounded hover:bg-blue-100 cursor-pointer"
          >
            {word}
          </li>
        ))}
      </ul>
    </div>
  );
};