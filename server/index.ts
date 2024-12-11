import express from 'express';
import type { Request, Response } from 'express';
import cors from 'cors';
import { LLMManager, DEFAULT_CONFIG } from './llm/config.js';

const app = express();
app.use(cors());
app.use(express.json());

const llmManager = new LLMManager();
let initializationError: Error | null = null;

const initializeLLM = async () => {
  try {
    await llmManager.initialize(DEFAULT_CONFIG);
    console.log("LLM initialized successfully");
    initializationError = null;
  } catch (error) {
    console.error("Failed to initialize LLM:", error);
    initializationError = error as Error;
  }
};

app.get("/api/status", (req: Request, res: Response) => {
  res.json({
    initialized: llmManager.isReady(),
    error: initializationError?.message
  });
});

app.post("/api/predict", async (req: Request, res: Response) => {
  if (!llmManager.isReady()) {
    return res.status(503).json({ 
      error: "LLM not initialized",
      details: initializationError?.message 
    });
  }

  try {
    const { text } = req.body;
    if (!text || typeof text !== 'string') {
      return res.status(400).json({ error: "Text is required and must be a string" });
    }
    
    const prediction = await llmManager.predictNextWord(text);
    if (!prediction) {
      return res.status(404).json({ error: "No prediction generated" });
    }
    
    res.json({ prediction });
  } catch (error) {
    console.error("Prediction error:", error);
    res.status(500).json({ 
      error: "Failed to generate prediction",
      details: error instanceof Error ? error.message : String(error)
    });
  }
});

process.on("SIGINT", async () => {
  await llmManager.cleanup();
  process.exit(0);
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, async () => {
  await initializeLLM();
  console.log(`Server running on port ${PORT}`);
});