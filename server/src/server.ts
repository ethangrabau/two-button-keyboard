import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';

const app = express();
app.use(cors());
app.use(express.json());

// Keyboard layout
const KEYBOARD = {
  L: ['Q', 'W', 'E', 'R', 'T', 'A', 'S', 'D', 'F', 'G', 'Z', 'X', 'C', 'V'],
  R: ['Y', 'U', 'I', 'O', 'P', 'H', 'J', 'K', 'L', 'B', 'N', 'M']
};

// Load word list with frequencies (from most common to least common)
const WORD_LIST = [
  // Most common chat words
  { word: 'hi', freq: 1000 },
  { word: 'hey', freq: 900 },
  { word: 'hello', freq: 800 },
  { word: 'thanks', freq: 750 },
  { word: 'yes', freq: 700 },
  { word: 'no', freq: 700 },
  { word: 'ok', freq: 650 },
  { word: 'good', freq: 600 },
  // Common English words
  { word: 'the', freq: 500 },
  { word: 'be', freq: 450 },
  { word: 'to', freq: 400 },
  { word: 'of', freq: 350 },
  { word: 'and', freq: 300 },
  // Add more words here...
];

// Pre-compute patterns for all words
const wordPatterns = new Map(
  WORD_LIST.map(({word}) => [
    word,
    word.toUpperCase().split('').map(char => 
      KEYBOARD.L.includes(char) ? 'L' : 'R'
    ).join('')
  ])
);

function getWordsForPattern(pattern: string, previousWords: string[] = []): string[] {
  const matches = WORD_LIST
    .filter(({word}) => wordPatterns.get(word)?.startsWith(pattern))
    .sort((a, b) => {
      // Prioritize based on frequency
      return b.freq - a.freq;
    })
    .map(({word}) => word)
    .slice(0, 5);

  return matches;
}

app.post('/api/predict', (req, res) => {
  try {
    const { currentText, pattern } = req.body;
    
    // Get previous words from current text
    const previousWords = currentText.trim().split(/\s+/);
    
    const startTime = performance.now();
    const predictions = getWordsForPattern(pattern, previousWords);
    const endTime = performance.now();

    res.json({
      predictions,
      latency: endTime - startTime
    });
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ 
      error: 'Prediction failed',
      predictions: [],
      latency: null
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});