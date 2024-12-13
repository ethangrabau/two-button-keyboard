import { getLlama, LlamaChatSession } from 'node-llama-cpp';
import path from 'path';

const PROJECT_ROOT = path.resolve('/Users/ethangrabau/Documents/two-button-keyboard');

export interface LLMConfig {
  modelPath: string;
  contextSize: number;
  batchSize: number;
  threads: number;
}

export const DEFAULT_CONFIG: LLMConfig = {
  modelPath: path.join(PROJECT_ROOT, "models", "tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf"),
  contextSize: 512,  // Reduced for simpler predictions
  batchSize: 512,
  threads: 4,
};

export class LLMManager {
  private model: any;
  private context: any;
  private session: LlamaChatSession | null = null;
  private isInitialized: boolean = false;
  
  async initialize(config: LLMConfig = DEFAULT_CONFIG) {
    try {
      console.log("Initializing LLM with model path:", config.modelPath);
      const llama = await getLlama();
      
      console.log("Loading model...");
      this.model = await llama.loadModel({
        modelPath: config.modelPath,
        gpuLayers: 0,  // CPU only for testing
        modelConfig: {
          vocabOnly: false
        }
      });
      
      console.log("Creating context...");
      this.context = await this.model.createContext({
        batchSize: config.batchSize,
        threads: config.threads
      });

      const sequence = await this.context.getSequence();
      this.session = new LlamaChatSession({ contextSequence: sequence });

      // Test simple prompt
      console.log("Testing simple prompt...");
      const testResponse = await this.session.prompt('Test', {
        maxTokens: 1,
        temperature: 0
      });
      console.log("Test response:", testResponse);
      
      this.isInitialized = true;
      console.log("LLM initialized successfully");
    } catch (error) {
      console.error("Failed to initialize LLM:", error);
      this.isInitialized = false;
      throw error;
    }
  }
  
  async predictNextWord(currentText: string): Promise<string> {
    if (!this.isInitialized || !this.session) {
      throw new Error("LLM not initialized");
    }

    try {
      // Simple prompt format
      const prompt = `${currentText.trim()} `;
      console.log("Sending prompt:", prompt);
      
      const response = await this.session.prompt(prompt, {
        maxTokens: 2,
        temperature: 0,
        topK: 1,
        frequencyPenalty: 0,
        presencePenalty: 0
      });
      
      console.log("Raw response:", response);
      
      if (!response) {
        console.log("Got empty response");
        return "the"; // Fallback
      }
      
      // Take first word only
      const word = response.trim().split(/[\s\n.,!?]+/)[0];
      console.log("Extracted word:", word);
      
      return word || "the";
      
    } catch (error) {
      console.error("Prediction error:", error);
      return "the"; // Fallback on error
    }
  }
  
  async cleanup() {
    try {
      this.session = null;
      
      if (this.context && typeof this.context.release === 'function') {
        await this.context.release();
      }
      
      if (this.model && typeof this.model.release === 'function') {
        await this.model.release();
      }
      
      this.isInitialized = false;
      console.log("Cleanup completed");
    } catch (error) {
      console.error("Cleanup warning:", error);
    }
  }

  isReady(): boolean {
    return this.isInitialized && this.session !== null;
  }
}