import express from 'express';
import cors from 'cors';
import { LlamaModel, LlamaContext, LlamaChatSession } from 'node-llama-cpp';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app = express();
app.use(cors());
app.use(express.json());

let model = null;
let context = null;
let session = null;

async function initializeModel() {
    try {
        const modelPath = path.join(__dirname, '..', 'models', 'tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf');
        
        model = new LlamaModel({
            modelPath: modelPath,
            contextSize: 512,
            batchSize: 512,
            gpuLayers: 0
        });

        context = new LlamaContext({ model });
        session = new LlamaChatSession({ context });
        
        console.log('Model initialized successfully');
        return true;
    } catch (error) {
        console.error('Failed to initialize model:', error);
        return false;
    }
}

// Initialize model when server starts
initializeModel();

app.post('/api/predict', async (req, res) => {
    try {
        const { currentText, pattern } = req.body;
        
        if (!model || !context || !session) {
            throw new Error('Model not initialized');
        }

        // Convert pattern to possible letters
        const keyboard = {
            'L': ['Q', 'W', 'E', 'R', 'T', 'A', 'S', 'D', 'F', 'G', 'Z', 'X', 'C', 'V'],
            'R': ['Y', 'U', 'I', 'O', 'P', 'H', 'J', 'K', 'L', 'B', 'N', 'M']
        };

        const possibleLetters = pattern
            .split('')
            .map(side => keyboard[side])
            .reduce((acc, letters) => 
                acc.length === 0 ? letters : 
                acc.filter(l1 => letters.some(l2 => l1 === l2)), 
            []);

        const prompt = `Complete this message with a word starting with one of these letters: ${possibleLetters.join(', ')}. Message: "${currentText}"`;

        const startTime = performance.now();
        const response = await session.prompt(prompt, {
            maxTokens: 10,
            temperature: 0.7,
            topK: 5,
            topP: 0.9,
            stop: ['\n', '.', '?', '!']
        });
        const endTime = performance.now();

        const predictions = response
            .split(/\s+/)
            .map(word => word.trim().toLowerCase())
            .filter(word => word.length > 0)
            .slice(0, 5);

        res.json({
            predictions,
            latency: endTime - startTime
        });
    } catch (error) {
        console.error('Error getting predictions:', error);
        res.status(500).json({
            error: error.message,
            predictions: [],
            latency: null
        });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});